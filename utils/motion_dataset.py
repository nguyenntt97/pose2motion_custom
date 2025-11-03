from pathlib import Path
from torch.utils.data import Dataset
import os
import sys
import numpy as np
import torch
from utils.Quaternions import Quaternions
from torch.nn.utils.rnn import pad_sequence

class MotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, data_path: Path, # /adult_dog/coyote
                 normalization: int = 1, 
                 data_augment: int = 0, 
                 window_size: int = 64,
                 step_size: int = 8):
        super(MotionData, self).__init__()
        self.data_augment: int = data_augment
        self.normalization: int = normalization
        self.data_path: Path = data_path
        self.window_size: int = window_size
        
        motion_path = data_path.parent / (data_path.name + ".npy")

        print('load from file {}'.format(motion_path))
        self.total_frame = 0
        self.std_bvh = MotionData._get_std_bvh(data_path)
        
        self.motion_length = []
        
        motions = np.load(motion_path, allow_pickle=True)  # list of (L, J, 3)
        self.data, self.window_idx = self._window(
                                    motions, 
                                    window_size=window_size, 
                                    step_size=step_size)

        if normalization == 1:
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.var = torch.var(self.data, (0, 2), keepdim=True)
            self.var = self.var ** (1/2)
            idx = self.var < 1e-5
            self.var[idx] = 1
            self.data = (self.data - self.mean) / self.var
        else:
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.mean.zero_()
            self.var = torch.ones_like(self.mean)

        # train_len = self.data.shape[0] * 95 // 100
        # self.test_set = self.data[train_len:, ...]
        # self.data = self.data[:train_len, ...]
        # self.data_reverse = torch.tensor(self.data.numpy()[..., ::-1].copy())

        # self.reset_length_flag = 0
        # self.virtual_length = 0
        print('Window count: {}, total frame (without downsampling): {}'.format(len(self), self.total_frame))

    def _get_std_bvh(bvh_parent: Path):
        # same sibling `std_bvhs` folder
        std_bvh_path = bvh_parent.parent / "std_bvhs" / f"{bvh_parent.name}.bvh"
        print("Standardized BVH path:", std_bvh_path)
        return std_bvh_path

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]
    
    def _window(self, motions, window_size: int, step_size: int):
        original_lengths = torch.tensor([len(pose) for pose in motions])
        
        padded_seqs = pad_sequence(
            [torch.from_numpy(m.astype(np.float32)) for m in motions],
            batch_first=True,
            padding_value=0.0
        ) # (N, L_max, J, 3), padded with zeros

        unfolded_windows = padded_seqs.unfold(1, window_size, step_size) # (N, L_max, J, 3) -> (N, num_windows, window_size, J, 3)
        orig_ids = torch.arange(len(motions)).unsqueeze(1).repeat(1, unfolded_windows.shape[1]) # (N * num_windows,)
        
        valid_num = ((original_lengths - window_size) // step_size) + 1
        valid_num = valid_num.clamp(min=0) # if seq_len < window_size
        padded_window_num = unfolded_windows.shape[1]
        
        mask = (torch.arange(padded_window_num)[None, :] < valid_num[:, None]) # (N, padded_window_num)
        valid_windows = unfolded_windows[mask] # (total_valid_windows, window_size, J, 3)
        orig_ids = orig_ids[mask] # (total_valid_windows,)
        
        return valid_windows, orig_ids # N, W, J, 3

    def subsample(self, motion):
        return motion[::2, :]

    def denormalize(self, motion):
        if self.args.normalization:
            if self.var.device != motion.device:
                self.var = self.var.to(motion.device)
                self.mean = self.mean.to(motion.device)
            ans = motion * self.var + self.mean
        else: ans = motion
        return ans
