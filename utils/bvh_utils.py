import re


class BvhNode:

    def __init__(self, value=[], parent=None):
        self.value = value
        self.children = []
        self.parent = parent
        if self.parent:
            self.parent.add_child(self)

    def add_child(self, item):
        item.parent = self
        self.children.append(item)

    def filter(self, key):
        for child in self.children:
            if child.value[0] == key:
                yield child

    def __iter__(self):
        for child in self.children:
            yield child

    def __getitem__(self, key):
        for child in self.children:
            for index, item in enumerate(child.value):
                if item == key:
                    if index + 1 >= len(child.value):
                        return None
                    else:
                        return child.value[index + 1:]
        raise IndexError('key {} not found'.format(key))

    def __repr__(self):
        return str(' '.join(self.value))

    @property
    def name(self):
        return self.value[1]


class Bvh:

    def __init__(self, data):
        self.data = data
        self.root = BvhNode()
        self.frames = []
        self.tokenize()

    def tokenize(self):
        first_round = []
        accumulator = ''
        for char in self.data:
            if char not in ('\n', '\r'):
                accumulator += char
            elif accumulator:
                    first_round.append(re.split('\\s+', accumulator.strip()))
                    accumulator = ''
        node_stack = [self.root]
        frame_time_found = False
        node = None
        for item in first_round:
            if frame_time_found:
                self.frames.append(item)
                continue
            key = item[0]
            if key == '{':
                node_stack.append(node)
            elif key == '}':
                node_stack.pop()
            else:
                node = BvhNode(item)
                node_stack[-1].add_child(node)
            if item[0] == 'Frame' and item[1] == 'Time:':
                frame_time_found = True

    def search(self, *items):
        found_nodes = []

        def check_children(node):
            if len(node.value) >= len(items):
                failed = False
                for index, item in enumerate(items):
                    if node.value[index] != item:
                        failed = True
                        break
                if not failed:
                    found_nodes.append(node)
            for child in node:
                check_children(child)
        check_children(self.root)
        return found_nodes

    def get_joints(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint)
            for child in joint.filter('JOINT'):
                iterate_joints(child)
        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def get_joints_names(self):
        joints = []

        def iterate_joints(joint, prefix=''):
            joints.append(prefix + joint.value[1])
            for child in joint.filter('End'):
                iterate_joints(child, prefix = joint.value[1] + "_")
                
            for child in joint.filter('JOINT'):
                iterate_joints(child)
            
        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def joint_direct_children(self, name):
        joint = self.get_joint(name)
        return [child for child in joint.filter('JOINT')]

    def get_joint_index(self, name):
        return self.get_joints().index(self.get_joint(name))

    def get_joint(self, name):
        found = self.search('ROOT', name)
        if not found:
            found = self.search('JOINT', name)
        if found:
            return found[0]
        raise LookupError(f'joint {name} not found')
    
    def get_joint_or_end(self, name):
        try:
            return self.get_joint(name)
        except LookupError:
            # should be End Site
            parent = name.split('_')[0]
            parent_joint = self.get_joint(parent)
            
            for child in parent_joint.filter('End'):
                return child

    def joint_offset(self, name):
        joint = self.get_joint_or_end(name)
        offset = joint['OFFSET']
        return (float(offset[0]), float(offset[1]), float(offset[2]))

    def joint_channels(self, name):
        try:
            joint = self.get_joint(name)
        except LookupError:
            # should be End Site -> no channels
            return []
        
        return joint['CHANNELS'][1:]

    def get_joint_channels_index(self, joint_name):
        index = 0
        for joint in self.get_joints():
            if joint.value[1] == joint_name:
                return index
            index += int(joint['CHANNELS'][0])
        raise LookupError('joint not found')

    def get_joint_channel_index(self, joint, channel):
        channels = self.joint_channels(joint)
        if channel in channels:
            channel_index = channels.index(channel)
        else:
            channel_index = -1
        return channel_index

    def frame_joint_channel(self, frame_index, joint, channel, value=None):
        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1 and value is not None:
            return value
        return float(self.frames[frame_index][joint_index + channel_index])

    def frame_joint_channels(self, frame_index, joint, channels, value=None):
        values = []
        joint_index = self.get_joint_channels_index(joint)
        for channel in channels:
            channel_index = self.get_joint_channel_index(joint, channel)
            if channel_index == -1 and value is not None:
                values.append(value)
            else:
                values.append(
                    float(
                        self.frames[frame_index][joint_index + channel_index]
                    )
                )
        return values

    def frames_joint_channels(self, joint, channels, value=None):
        all_frames = []
        joint_index = self.get_joint_channels_index(joint)
        for frame in self.frames:
            values = []
            for channel in channels:
                channel_index = self.get_joint_channel_index(joint, channel)
                if channel_index == -1 and value is not None:
                    values.append(value)
                else:
                    values.append(
                        float(frame[joint_index + channel_index]))
            all_frames.append(values)
        return all_frames

    def joint_parent(self, name):
        joint = self.get_joint_or_end(name)
        if joint.parent == self.root:
            return None
        return joint.parent

    def joint_parent_index(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return -1
        return self.get_joints().index(joint.parent)

    @property
    def nframes(self):
        try:
            return int(next(self.root.filter('Frames:')).value[1])
        except StopIteration:
            raise LookupError('number of frames not found')

    @property
    def frame_time(self):
        try:
            return float(next(self.root.filter('Frame')).value[2])
        except StopIteration:
            raise LookupError('frame time not found')


import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D

def _separate_angles(frames, joints, joints_saved_channels):

    frame_i = 0
    joints_saved_angles = {}
    get_channels = []
    for joint in joints:
        _saved_channels = joints_saved_channels[joint]

        saved_rotations = []
        for chan in _saved_channels:
            if chan.lower().find('rotation') != -1:
                saved_rotations.append(chan)
                get_channels.append(frame_i)

            frame_i += 1
        joints_saved_angles[joint] = saved_rotations

    joints_rotations = frames[:,get_channels]

    return joints_rotations, joints_saved_angles

def _separate_positions(frames, joints, joints_saved_channels):

    frame_i = 0
    joints_saved_positions = {}
    get_channels = []
    for joint in joints:
        _saved_channels = joints_saved_channels[joint]

        saved_positions = []
        for chan in _saved_channels:
            if chan.lower().find('position') != -1:
                saved_positions.append(chan)
                get_channels.append(frame_i)

            frame_i += 1
        joints_saved_positions[joint] = saved_positions


    if len(get_channels) == 3*len(joints):
        #print('all joints have saved positions')
        return frames[:,get_channels], joints_saved_positions

    #no positions saved for the joints or only some are saved.
    else:
        return np.array([]), joints_saved_positions

    pass

def clean_bvh_joints_name(text_str: str):
    """
        Change _ to - in joint names to avoid issues later on.
    """
    # capture all `JOINT <name>` and `ROOT <name>` occurrences
    pattern = r'(JOINT|ROOT)\s+([^\s]+)'
    def replace_underscores(match):
        joint_type = match.group(1)
        joint_name = match.group(2).replace('_', '-')
        return f"{joint_type} {joint_name}"
    cleaned_str = re.sub(pattern, replace_underscores, text_str)
    return cleaned_str 

SKELETON_1_JOINTS = ['SKEL-Pelvis', 'SKEL-Spine0', 'SKEL-Spine1', 'SKEL-L-Clavicle', 'SKEL-L-UpperArm', 'SKEL-L-Forearm', 'SKEL-L-Hand', 'SKEL-L-Finger00', 'SKEL-L-Finger01', 'SKEL-L-Finger01_Site', 'SKEL-R-Clavicle', 'SKEL-R-UpperArm', 'SKEL-R-Forearm', 'SKEL-R-Hand', 'SKEL-R-Finger00', 'SKEL-R-Finger01', 'SKEL-R-Finger01_Site', 'SKEL-Spine2', 'SKEL-Spine3', 'SKEL-Neck1', 'SKEL-Neck2', 'SKEL-Head', 'SKEL-Head_Site', 'SKEL-L-Thigh', 'SKEL-L-Calf', 'SKEL-L-Foot', 'SKEL-L-Toe0', 'SKEL-L-Toe1', 'SKEL-L-Toe1_Site', 'SKEL-R-Thigh', 'SKEL-R-Calf', 'SKEL-R-Foot', 'SKEL-R-Toe0', 'SKEL-R-Toe1', 'SKEL-R-Toe1_Site'] #labrador and coyote

all_skeletons = [SKELETON_1_JOINTS]

def ProcessBVH(filename, skeleton_id: int = 0):

    with open(filename) as f:
        mocap = Bvh(clean_bvh_joints_name(f.read()))

    #get the names of the joints
    joints = mocap.get_joints_names()
    joints = [j for j in joints if j in all_skeletons[skeleton_id]]

    #this contains all of the frames data.
    frames = np.array(mocap.frames).astype('float32')

    #determine the structure of the skeleton and how the data was saved
    joints_offsets = {}
    joints_hierarchy = {}
    joints_saved_channels = {}
    for joint in joints:
        #get offsets. This is the length of skeleton body parts
        joints_offsets[joint] = np.array(mocap.joint_offset(joint))

        #Some bvh files save only rotation channels while others also save positions.
        #the order of rotation is important
        joints_saved_channels[joint] = mocap.joint_channels(joint)

        #determine the hierarcy of each joint.
        joint_hierarchy = []
        parent_joint = joint
        while True:
            parent_name = mocap.joint_parent(parent_joint)
            if parent_name == None:break

            joint_hierarchy.append(parent_name.name)
            parent_joint = parent_name.name

        joints_hierarchy[joint] = joint_hierarchy

    #seprate the rotation angles and the positions of joints
    joints_rotations, joints_saved_angles = _separate_angles(frames, joints, joints_saved_channels)
    joints_positions, joints_saved_positions = _separate_positions(frames, joints, joints_saved_channels)

    #root positions are always saved
    root_positions = frames[:, 0:3]

    return [joints, joints_offsets, joints_hierarchy, root_positions, joints_rotations, joints_saved_angles, joints_positions, joints_saved_positions]

#rotation matrices
def Rx(ang, in_radians = False):
    if in_radians == False:
        ang = np.radians(ang)

    Rot_Mat = np.array([
        [1, 0, 0],
        [0, np.cos(ang), -1*np.sin(ang)],
        [0, np.sin(ang),    np.cos(ang)]
    ])
    return Rot_Mat

def Ry(ang, in_radians = False):
    if in_radians == False:
        ang = np.radians(ang)

    Rot_Mat = np.array([
        [np.cos(ang), 0, np.sin(ang)],
        [0, 1, 0],
        [-1*np.sin(ang), 0, np.cos(ang)]
    ])
    return Rot_Mat

def Rz(ang, in_radians = False):
    if in_radians == False:
        ang = np.radians(ang)

    Rot_Mat = np.array([
        [np.cos(ang), -1*np.sin(ang), 0],
        [np.sin(ang), np.cos(ang), 0],
        [0, 0, 1]
    ])
    return Rot_Mat

#the rotation matrices need to be chained according to the order in the file
def _get_rotation_chain(joint_channels, joint_rotations):

    #the rotation matrices are constructed in the order given in the file
    Rot_Mat =  np.array([[1,0,0],[0,1,0],[0,0,1]])#identity matrix 3x3
    order = ''
    index = 0
    for chan in joint_channels: #if file saves xyz ordered rotations, then rotation matrix must be chained as R_x @ R_y @ R_z
        if chan[0].lower() == 'x':
            Rot_Mat = Rot_Mat @ Rx(joint_rotations[index])
            order += 'x'

        elif chan[0].lower() == 'y':
            Rot_Mat = Rot_Mat @ Ry(joint_rotations[index])
            order += 'y'

        elif chan[0].lower() == 'z':
            Rot_Mat = Rot_Mat @ Rz(joint_rotations[index])
            order += 'z'
        index += 1
    #print(order)
    return Rot_Mat

#Here root position is used as local coordinate origin.
def _calculate_frame_joint_positions_in_local_space(joints, joints_offsets, frame_joints_rotations, joints_saved_angles, joints_hierarchy):

    local_positions = {}

    for joint in joints:

        #ignore root joint and set local coordinate to (0,0,0)
        if joint == joints[0]:
            pos = [0, 0, 0]
            Rot = np.eye(3) #identity matrix
            
            joint_pos = joints_offsets[joint]
            joint_pos = Rot @ joint_pos
            pos = pos + joint_pos
            local_positions[joint] = pos
            continue

        connected_joints = joints_hierarchy[joint]
        connected_joints = connected_joints[::-1]
        connected_joints.append(joint) #this contains the chain of joints that finally end with the current joint that we want the coordinate of.
        Rot = np.eye(3)
        pos = [0,0,0]
        for i, con_joint in enumerate(connected_joints):
            if i == 0:
                pass
            else:
                parent_joint = connected_joints[i - 1]
                Rot = Rot @ _get_rotation_chain(joints_saved_angles[parent_joint], frame_joints_rotations[parent_joint])
            joint_pos = joints_offsets[con_joint]
            joint_pos = Rot @ joint_pos
            pos = pos + joint_pos

        local_positions[joint] = pos

    return local_positions

def _calculate_frame_joint_positions_in_world_space(local_positions, root_position, root_rotation, saved_angles):

    world_pos = {}
    for joint in local_positions:
        pos = local_positions[joint]

        # Rot = _get_rotation_chain(saved_angles, root_rotation)
        # pos = Rot @ pos

        pos = np.array(root_position) + pos
        world_pos[joint] = pos

    return world_pos

def _calculate_t_pose(joints, joints_offsets, joints_hierarchy):

    t_pose_local_positions = []

    for joint in joints:

        #ignore root joint and set local coordinate to (0,0,0)
        if joint == joints[0]:
            pos = [0, 0, 0]
            
            joint_pos = joints_offsets[joint]
            pos = pos + joint_pos
            t_pose_local_positions.append(pos)
            continue

        connected_joints = joints_hierarchy[joint]
        connected_joints = connected_joints[::-1]
        connected_joints.append(joint) #this contains the chain of joints that finally end with the current joint that we want the coordinate of.
        
        pos = [0,0,0]
        for i, con_joint in enumerate(connected_joints):
            joint_pos = joints_offsets[con_joint]
            pos = pos + joint_pos

        t_pose_local_positions.append(pos)

    return np.array(t_pose_local_positions) # dimensions: n_joints x 3