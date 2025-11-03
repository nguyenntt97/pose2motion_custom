import os
from pathlib import Path
import shutil
import numpy as np
import copy
import sys
from utils.bvh_parser import BVH_file
from utils.motion_dataset import MotionData
from utils.option_parser import get_args, try_mkdir
import torch

def collect_bvh(data_path: Path, character: Path, files: list[Path]):
    print('begin {}'.format(character))
    motions = []

    for i, motion in enumerate(files):
        print(f"Processing file {i+1}/{len(files)}: {motion.name}")
        file = BVH_file(motion)
        new_motion = file.to_tensor().to(dtype=torch.float32).permute((1, 0)).numpy()
        motions.append(new_motion)
        print(f"New motion {new_motion.shape}")

    save_file: Path = data_path / f"{character.name}.npy"

    np.save(save_file, np.array(motions, dtype=object))
    print('Npy file saved at {}'.format(save_file))


def write_statistics(character: Path, path):
    args = get_args()
    new_args = copy.copy(args)
    new_args.data_augment = 0
    new_args.dataset = character.parent / f"{character.name}.npy"

    dataset = MotionData(new_args)

    mean = dataset.mean
    var = dataset.var
    mean = mean.cpu().numpy()[0, ...]
    var = var.cpu().numpy()[0, ...]

    np.save(path / '{}_mean.npy'.format(character.name), mean)
    np.save(path / '{}_var.npy'.format(character.name), var)


def copy_std_bvh(data_path: Path, character: Path, files: list[Path]):
    """
    copy an arbitrary bvh file as a static information (skeleton's offset) reference
    """
    # cmd = 'cp \"{}\" ./datasets/Mixamo/std_bvhs/{}.bvh'.format(data_path + character + '/' + files[0], character)
    # os.system(cmd)
    
    # copy the first bvh file as standard bvh
    shutil.copy(files[0], data_path / 'std_bvhs' / f"{character.name}.bvh")


if __name__ == '__main__':
    prefix = Path('./data/aino_zoo/adult_dog_new')
    characters = list([d for d in prefix.iterdir() if d.is_dir()])
    print(characters)
    for character in characters:
        if "std_bvhs" in character.name or "mean_var" in character.name:
            print(f"Removing directory from processing: {character}")
            characters.remove(character)
    
    # if 'std_bvhs' in characters: characters.remove('std_bvhs')
    # if 'mean_var' in characters: characters.remove('mean_var')

    try_mkdir(os.path.join(prefix, 'std_bvhs'))
    try_mkdir(os.path.join(prefix, 'mean_var'))

    for character in characters:
        files = list(character.glob('*.bvh'))

        collect_bvh(prefix, character, files)
        copy_std_bvh(prefix, character, files)
        
        (prefix / 'mean_var').mkdir(exist_ok=True)
        write_statistics(character, prefix / 'mean_var')
