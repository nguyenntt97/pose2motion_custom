import sys
import cv2
import numpy as np
from tensorboard import data
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple
import numpy.typing as npt

from pathlib import Path
from bvh_utils import ProcessBVH, _calculate_frame_joint_positions_in_local_space, _calculate_frame_joint_positions_in_world_space
import numpy as np

def super_resolute_motion(sequence, target_fps=250, cur_fps=30):
    """Super-resolute motion sequence using spline interpolation.

    Args:
        sequence (_type_): (T, N, 3) array where T is the number of frames,
            N is the number of joints, and 3 represents the x, y, z coordinates
        target_fps (int, optional): _description_. Defaults to 250.
        cur_fps (int, optional): _description_. Defaults to 30.
    """
    
    # Resample the motion sequences with spline interpolation
    from scipy.interpolate import RBFInterpolator
    time_points = np.arange(0, len(sequence) / cur_fps, 1 / cur_fps)
    new_time_points = np.arange(0, len(sequence) / cur_fps, 1 / target_fps)
    new_sequence = np.zeros((len(new_time_points), sequence.shape[1], 3))

    for joint_idx in range(sequence.shape[1]):
        cs = RBFInterpolator(time_points[:, np.newaxis], sequence[:, joint_idx, :], kernel='thin_plate_spline')
        new_sequence[:, joint_idx, :] = cs(new_time_points[:, np.newaxis])

    return new_sequence

def convert_bvh_to_csv(test_file: Path, output_dir: Path):
    data = ProcessBVH(str(test_file))
    
    joints = data[0]
    joints_offsets = data[1]
    joints_hierarchy = data[2]
    root_positions = data[3]
    joints_rotations = data[4]
    joints_saved_angles = data[5]
    joints_positions = data[6]
    joints_saved_positions = data[7]
    
    frame_joints_rotations = {en:[] for en in joints}

    joints_df = []

    for i in tqdm(range(0, len(joints_rotations))):
        # visualize the joints in the first motion sequence 
        frame_data = joints_rotations[i]
        joint_index = 0
        for joint in joints:
            if "_Site" in joint: 
                frame_joints_rotations[joint] = [0, 0, 0] # end sites have no rotation
                continue #skip end sites
            
            frame_joints_rotations[joint] = frame_data[joint_index:joint_index+3]
            joint_index += 3

        local_pos = _calculate_frame_joint_positions_in_local_space(joints, joints_offsets, frame_joints_rotations, joints_saved_angles, joints_hierarchy)
        world_pos = _calculate_frame_joint_positions_in_world_space(local_pos, root_positions[i], frame_joints_rotations[joints[0]], joints_saved_angles[joints[0]])
        
        cur_frame = []
        for joint in joints:
            cur_frame.append([world_pos[joint][0], world_pos[joint][2], world_pos[joint][1]])
            if joint == joints[0]: continue #skip root joint
            

        joints_df.append(cur_frame)
    
    # Joints ids
    for i, joint in enumerate(joints):
        print(f"{i}: {joint}")
    
    test_data = np.array(joints_df)[:, :, [0, 2, 1]] # (X, Y, Z) -> (X, Z, Y) PyBullet convention
    N, J, C = test_data.shape
    print(f"Converted {test_file.name} to CSV with shape: {test_data.shape}")
    test_data = normalize_data(test_data)  # Normalize the data
    # test_data = super_resolute_motion(test_data, target_fps=200, cur_fps=60)  # Super-resolute the motion sequence
    
    # Save the data to a CSV file and keep 5 decimal places
    output_file = output_dir / f"{test_file.stem}_converted.txt"
    np.savetxt(output_file, test_data.reshape(-1, J * C), delimiter=",", fmt='%.5f')
    
def normalize_data(data: npt.NDArray):
    """Get ROOT joint X, Y's starting position as 0, smallest Z as 0, and scale 1:100

    Args:
        data (npt.NDArray): (T, J, 3) array of joint positions
    """
    root_pos = data[0, 0, [0, 2]]  # Get the root joint's X, Y position
    min_z = data[:, :, 1].min()  # Get the minimum Z position across all joints and frames

    # Center the data
    data[:, :, 0] -= root_pos[0]  # Center X
    data[:, :, 2] -= root_pos[1]  # Center Y
    data[:, :, 1] -= min_z  # Center Z

    # Scale the data
    data = data / 100.0  # Scale all joint positions by 1:100

    return data

if __name__ == "__main__":
    # arguments
    import argparse
    parser = argparse.ArgumentParser(description="Convert BVH file to CSV format.")
    parser.add_argument("--test_file", type=Path, required=True, help="Path to the BVH file to convert.")
    parser.add_argument("--output_dir", type=Path, help="Directory to save the converted CSV file.")
    
    args = parser.parse_args()
    test_file = args.test_file
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = test_file.parent

    convert_bvh_to_csv(test_file, output_dir)