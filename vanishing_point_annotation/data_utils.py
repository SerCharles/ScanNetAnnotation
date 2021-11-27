"""The data util functions of vanishing point annotation
"""
import os
import numpy as np
from PIL import Image
from numpy.lib import utils
import utils

def load_image(file_name):
    """load a RGB image
        H: the height of the picture
        W: the width of the picture
        
    Args:
        file_name [str]: [the place of the RGB image]

    Returns:
        pic_array [numpy float array], [H * W * 3(RGB) or H * W(grey scale)]: [the PIL data of the image]
    """
    fp = open(file_name, 'rb')
    pic = Image.open(fp)
    pic_array = np.array(pic)
    fp.close()
    return pic_array

def load_intrinsic(file_name):
    """load an intrinsic file of ScanNet

    Args:
        file_name [str]: [the place of the intrinsic file]

    Returns:
        intrinsic [numpy float array], [3 * 3]: [the intrinsic array]
    """
    intrinsic = np.zeros((3, 3), dtype = float)
    intrinsic[2][2] = 1.0

    f = open(file_name, 'r')
    lines = f.read().split('\n')
    for line in lines: 
        words = line.split()
        if len(words) > 0:
            if words[0] == 'm_calibrationColorIntrinsic':
                intrinsic[0][0] = float(words[2])
                intrinsic[1][1] = float(words[7])
                intrinsic[2][0] = float(words[4])
                intrinsic[2][1] = float(words[8])

    f.close()
    return intrinsic

def load_extrinsic(file_name):
    """load an extrinsic file of ScanNet

    Args:
        file_name [str]: [the name of the extrinsic file]

    Returns:
        pose [numpy float array], [4 * 4]: [the extrinsic numpy array]
    """
    pose = np.zeros((4, 4), dtype = np.float32)
    f = open(file_name, 'r')
    lines = f.read().split('\n')
    
    for i in range(4):
        words = lines[i].split()
        for j in range(4):
            word = float(words[j])
            pose[i][j] = word

    pose = pose.astype(np.float32)
    f.close()
    return pose


def load_data(base_dir, scene_id, id):
    """Load the data of picture of one group
        H: the height of the picture
        W: the width of the picture
        
    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the scene id to be handled]
        id [int]: [the id of the picture]
        
    Return:
        intrinsic [numpy float array], [3 * 3]: [the intrinsic of the picture]
        extrinsic [numpy float array], [4 * 4]: [the extrinsic of the picture]
        layout_seg [numpy int array], [H * W]: [the layout segmentation label of the picture]
    """
    base_name = scene_id + '_' + str(id)
    intrinsic_name = os.path.join(base_dir, scene_id, '_info.txt')
    extrinsic_name = os.path.join(base_dir, scene_id, 'pose', base_name + ".txt")
    layout_seg_name = os.path.join(base_dir, scene_id, 'layout_seg', base_name + '.png')
    intrinsic = load_intrinsic(intrinsic_name)
    extrinsic = load_extrinsic(extrinsic_name)
    layout_seg = load_image(layout_seg_name).astype(np.int)
    
    return intrinsic, extrinsic, layout_seg

def save_boundaries(base_dir, scene_id, id, vanishing_point, boundary_angles, boundary_segs, wall_seg):
    """Save the boundary data

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the scene id to be handled]
        id [int]: [the id of the picture]        
        vanishing_point [numpy float array], [2]: [the vanishing point of the picture, (y, x)]    
        boundary_angles [numpy float array], [M]: [the absolute angles of the boundaries]
        boundary_segs [numpy int array], [M * 2]: [the left and right seg numbers of the boundaries]
        wall_seg [numpy bool array], [H * W]: [the pixels whose backgrounds are walls]
    """
    save_dir_npy = os.path.join(base_dir, scene_id, 'boundary')
    if not os.path.exists(save_dir_npy):
        os.mkdir(save_dir_npy)
    save_place_npy = os.path.join(save_dir_npy, scene_id + '_' + str(id) + '.npz')
    np.savez(save_place_npy, vanishing_point=vanishing_point, boundary_angles=boundary_angles, boundary_segs=boundary_segs)
    save_dir_seg = os.path.join(base_dir, scene_id, 'wall_seg')
    if not os.path.exists(save_dir_seg):
        os.mkdir(save_dir_seg)
    save_place_seg = os.path.join(save_dir_seg, scene_id + '_' + str(id) + '.png')
    picture_boundary = wall_seg.astype(np.uint16)
    picture_boundary = Image.fromarray(picture_boundary)
    picture_boundary.save(save_place_seg)


def visualize_boundaries(save_dir, base_name, vanishing_point, boundary_probability_per_pixel, layout_seg):
    """Visualize the boundaries

    Args:
        save_dir [string]: [the base save directory of the data]
        base_name [string]: [the base name of the picture]
        vanishing_point [numpy float array], [2]: [the vanishing point of the picture, (y, x)]    
        boundary_probability_per_pixel [numpy float array], [H * W]: [whether each pixels are boundaries]
        layout_seg [numpy int array], [H * W]: [the layout segmentation of the picture]
    """
    H, W = layout_seg.shape
    full_save_dir_seg = os.path.join(save_dir, base_name + '_seg.png')
    full_save_dir_prob = os.path.join(save_dir, base_name + '_prob.png')
    final_color_seg = (layout_seg * 3000).astype(np.uint16)
    final_color_prob = (boundary_probability_per_pixel * 60000).astype(np.uint16)
    picture_seg = Image.fromarray(final_color_seg)
    picture_seg.save(full_save_dir_seg)
    picture_prob = Image.fromarray(final_color_prob)
    picture_prob.save(full_save_dir_prob)
    print('Written', base_name)
        
    
        
    