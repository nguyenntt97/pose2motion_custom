from dataclasses import dataclass
from pathlib import Path
from sympy.logic.inference import valid
from sympy.physics.mechanics import joint
import torch
from numpy.typing import NDArray
import numpy as np
from utils.bvh_utils import ProcessBVH, _calculate_frame_joint_positions_in_local_space, _calculate_frame_joint_positions_in_world_space, _calculate_t_pose
from torch.nn.utils.rnn import pad_sequence

import torch
import numpy as np

from os import path as osp
import random
import glob as glob
from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms
from data_util import *

@dataclass
class AinoZooInfo:
    joint_parents_idx_full: NDArray[np.int32] # full joint parents idx
    joint_parents_idx: NDArray[np.int32]
    joint_offsets: NDArray[np.float32] # t-pose offsets
    end_sites: NDArray[np.int32]
    t_pose: NDArray[np.float32] # joint positions in t-pose (J, 3)
    joint_names: list[str]
    root_tr: NDArray[np.float32] # root translation (3,)
    root_R: NDArray[np.float32] # root rotation (3, 3)
    pose_virtual_index: NDArray[np.int32]  # indices of virtual joints

@dataclass
class ExtendedAinoZooInfo(AinoZooInfo): # extended info from AinoZooInfo
    foot_height: NDArray[np.float32] # foot height (1,)
    legs_lengths: NDArray[np.float32] # heights (N,)
    foot_index: NDArray[np.int32] # foot joint indices
    height: float
    z_range: list[float] # z min max
    x_range: list[float] # x min max
    y_range: list[float] # y min max
    
    virtual_mask: NDArray[np.float32] # mask for virtual joints
    pose_virtual_val: torch.Tensor = None # values of virtual joints
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.virtual_mask = np.ones((1, len(self.joint_names), 3), dtype=np.float32)
        self.virtual_mask[0, self.pose_virtual_index, :] = 0.0
        
        self.foot_height = self.t_pose[self.end_sites[3], 1] # y-axis is up
        self.legs_lengths = self.get_height(self.end_sites, self.joint_parents_idx_full, self.joint_offsets)
        self.foot_index = [0, 1] # TODO: check which joints are feet
        self.height = self.legs_lengths[0] + self.legs_lengths[2] # Why?
        
        self.z_range = [np.min(self.t_pose[:, 2]), np.max(self.t_pose[:, 2])]
        self.x_range = [np.min(self.t_pose[:, 0]), np.max(self.t_pose[:, 0])]
        self.y_range = [np.min(self.t_pose[:, 1]), np.max(self.t_pose[:, 1])]
        
    def get_height(self, end_list,parent_list,offset):
        height_list = []
        for i in end_list:
            p = parent_list[i]
            height = 0
            while p!=0:
                height += np.dot(offset[p],offset[p])**0.5
                p = parent_list[p]
            height_list.append(height)
        return height_list

class AinoZoo(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path, class_names: list[str], 
                 use_velo_virtual_node: bool = True,
                 use_gt_velocity: bool = False):
        self.dataset_path = dataset_path
        self.class_names = class_names # adult, child, etc.
        self.class2idx = {name: idx for idx, name in enumerate(class_names)}
        self.total_frames: int = 0
        
        self.use_velo_virtual_node = use_velo_virtual_node # B = False; A = True
        self.use_gt_velocity = use_gt_velocity # B = False; A = True
        
        self._load_dataset()
        self.foot_index = self.infos[0].foot_index
        self.height = self.infos[0].height
        self.foot_height = self.infos[0].foot_height
        self.threshold = 0.003 # TODO: Why?
        self.animal = "dog" # TODO: for contact loss? and more..?
        self.num_joints = self.poses[0].shape[1]


    def __len__(self):
        return len(self.windowed_poses)
    
    def _load_bvh(self, bvh_path: str):
        data = ProcessBVH(bvh_path)
        return {
            "joints": data[0],
            "joints_offsets": data[1],
            "joints_hierarchy": data[2],
            "root_positions": data[3],
            "joints_rotations": data[4],
            "joints_saved_angles": data[5],
            "joints_positions": data[6],
            "joints_saved_positions": data[7]
        }
        
    def _load_dataset(self):
        self.poses: list[NDArray[np.float32]] = []
        self.infos: list[ExtendedAinoZooInfo] = []
        self.labels: list[int] = []
        self.file_paths: list[str] = []
        self.joint2idx: dict[str, int] = {}

        sub_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        # sub_dir is label
        total_N: int = 0
        for sub_dir in sub_dirs:
            label = sub_dir.name
            if label not in self.class_names:
                continue
            
            bvh_files = list(sub_dir.glob("*/*.bvh"))
            for f in bvh_files:
                breed: str = f.parent.name
                if breed not in ["coyote", "labrador"]:
                    continue
                
                bvh_data = self._load_bvh(str(f))
                
                if len(self.joint2idx) == 0:
                    for idx, j in enumerate(bvh_data["joints"]):
                        self.joint2idx[j] = idx
                
                new_rot: NDArray[np.float32] = bvh_data["joints_rotations"]
                N, _ = new_rot.shape
                total_N += N
                
                new_rot = new_rot.reshape(N, -1, 3) # (N, J, 3)
                
                poses = torch.from_numpy(new_rot)
                if self.use_velo_virtual_node:
                    # add root velocity as virtual joint
                    root_velo = torch.zeros_like(poses[:, 0, :]) # (N, 1, 3)
                    gt_root_velo = bvh_data["root_positions"][1:] - bvh_data["root_positions"][:-1]
                    if self.use_gt_velocity:
                        root_velo[..., :3] = gt_root_velo[:]
                    
                    # attach as child of root
                    poses = torch.cat([poses, root_velo.unsqueeze(1)], dim=1) # (N, J+1, 3)
                    bvh_data["joints"].append("root_velocity_node")
                    bvh_data["joints_hierarchy"]["root_velocity_node"] = [bvh_data["joints"][0]] # root is parent
                    bvh_data["joints_offsets"]["root_velocity_node"] = [0.0, 0.0, 0.0]
                    # self.root_velos.append(root_velo)
                    
                    # update joint2idx
                    if "root_velocity_node" not in self.joint2idx:
                        self.joint2idx["root_velocity_node"] = len(self.joint2idx)

                pose_virtual_index = torch.where(((torch.abs(poses[0,:] - poses[20,:])) < 1e-8).sum(axis=1) == 3) # euler angles unchanged
                if self.use_velo_virtual_node:
                    pose_virtual_index = [pose_virtual_index[0][:-1]] # exclude the last virtual joint (root velocity)
                
                joint_parents_idx, joint_parents_idx_full = self.get_joint_parents_idx(
                    bvh_data["joints_hierarchy"], bvh_data["joints"])
                
                new_info_raw = ExtendedAinoZooInfo(
                    joint_parents_idx=joint_parents_idx,
                    joint_parents_idx_full=joint_parents_idx_full,
                    joint_offsets={self.joint2idx[k]: v for k, v in bvh_data["joints_offsets"].items()},
                    end_sites=[
                        self.joint2idx[k] for k in bvh_data["joints_hierarchy"].keys() 
                        if "_site" in k.lower()
                    ],
                    t_pose=_calculate_t_pose(
                        bvh_data["joints"], 
                        bvh_data["joints_offsets"], 
                        bvh_data["joints_hierarchy"]),
                    joint_names=bvh_data["joints"],
                    root_tr=bvh_data["root_positions"],
                    root_R=bvh_data["joints_rotations"][:, 0:3], # assuming first 3 are root rotations,
                    pose_virtual_index=pose_virtual_index,
                    
                )

                self.poses.append(self._to_rotation_6d(poses)) # (N, J, 3) -> (N, J, 6)
                self.labels.append(self.class2idx[label])
                self.file_paths.append(str(f))
                
                pose_virtual_val = torch.zeros_like(poses[0])
                pose_virtual_val[pose_virtual_index] = poses[0][pose_virtual_index]
                new_info_raw.pose_virtual_val = pose_virtual_val
                
                self.infos.append(new_info_raw)
                
        self.total_frames = total_N
        
        # windowing
        window_size = 64
        step_size = 4
        self.windowed_poses, self.windowed_ids = self._window(window_size, step_size)
        
        # normalization
        self.pose_mean, self.pose_std = self.get_normalize()
        self.windowed_poses = (self.windowed_poses - self.pose_mean) / self.pose_std
        self.joint_parents_idx = self.infos[0].joint_parents_idx
        
    def get_joint_parents_idx(self, hierarchy: dict[str, list[str]], joints: list[str]):
        joint_parents_idx = []
        joint_parents_idx_full = []
        for _, j in enumerate(joints):
            pa = self.joint2idx[hierarchy[j][0]] if len(hierarchy[j]) > 0 else 0
            joint_parents_idx_full.append(pa)
            if "_site" not in j.lower():
                joint_parents_idx.append(pa)
        return joint_parents_idx, joint_parents_idx_full
        
    def _window(self, window_size: int, step_size: int):
        original_lengths = torch.tensor([len(pose) for pose in self.poses])
        
        padded_seqs = pad_sequence(
            self.poses,
            batch_first=True,
            padding_value=0.0
        ) # (N, L_max, J, 3), padded with zeros

        unfolded_windows = padded_seqs.unfold(1, window_size, step_size) # (N, L_max, J, 3) -> (N, num_windows, window_size, J, 3)
        orig_ids = torch.arange(len(self.poses)).unsqueeze(1).repeat(1, unfolded_windows.shape[1]) # (N * num_windows,)
        
        valid_num = ((original_lengths - window_size) // step_size) + 1
        valid_num = valid_num.clamp(min=0) # if seq_len < window_size
        padded_window_num = unfolded_windows.shape[1]
        
        mask = (torch.arange(padded_window_num)[None, :] < valid_num[:, None]) # (N, padded_window_num)
        valid_windows = unfolded_windows[mask] # (total_valid_windows, window_size, J, 3)
        orig_ids = orig_ids[mask] # (total_valid_windows,)
        
        return valid_windows, orig_ids # N, W, J, 3
        
    def get_normalize(self, eps=1e-5):
        mean = torch.mean(self.windowed_poses, dim=(0), keepdim=True) # (J, 3)
        std = torch.std(self.windowed_poses, dim=(0), keepdim=True) + eps # (J, 3)
        return mean, std

    def __getitem__(self, idx):
        """Get item from dataset by index. Pose and Info"""
        # pose = self.poses[idx]
        # info = self.infos[idx]
        # label = self.labels[idx]
        # file_path = self.file_paths[idx]
        windows = self.windowed_poses
        # window_ids = self.windowed_ids
        # window_info = self.infos[window_ids[idx]]
        
        return windows[idx], idx

    def _to_rotation_6d(self, pose): # pose 3D euler (B, J, 3)
        B, J, _ = pose.shape
        rot_matrix = pytorch3d.transforms.euler_angles_to_matrix(
            pose.reshape(-1, 3), convention="XYZ"
        ) # (B*J, 3, 3)
        
        pose6d = pytorch3d.transforms.matrix_to_rotation_6d(
            rot_matrix
        ).reshape(B, J, 6)
        
        # b = pose.shape[0]
        # if type(pose) == np.ndarray:
        #     pose = torch.from_numpy(pose).float()
        # pose = pytorch3d.transforms.matrix_to_rotation_6d(pose.reshape(b, -1, 3, 3))
        return pose6d
