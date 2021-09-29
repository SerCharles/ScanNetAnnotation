"""The data util functions of vanishing point annotation
"""
import os
import numpy as np
from PIL import Image


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

def save_annotation_result(full_save_dir, vanishing_y, vanishing_x, whether_ceilings, whether_floors, whether_walls, whether_boundaries, ceiling_places, floor_places):
    """Save the annotation result

    Args:
        full_save_dir [string]: [the full place to save the numpy arraus]
        vanishing_y [float]: [the y of the vanishing point of the picture]    
        vanishing_x [float]: [the x of the vanishing point of the picture] 
        whether_ceilings [numpy boolean array], [(2 * W)]: [whether the lines have ceiling]
        whether_floors [numpy boolean array], [(2 * W)]: [whether the lines have floor]
        whether_walls [numpy boolean array], [(2 * W)]: [whether the lines have wall]
        whether_boundaries [numpy boolean array], [(2 * W)]: [whether the lines are wall-wall boundaries]
        ceiling_places [numpy float array], [2 * (2 * W)]: [the ceiling place of each line, (y, x)]
        floor_places [numpy float array], [2 * (2 * W)]: [the floor place of each line, (y, x)]
    """
    vanishing_point = np.zeros((2), dtype=np.float32)
    vanishing_point[0] = vanishing_y
    vanishing_point[1] = vanishing_x
    np.savez(full_save_dir, vanishing_point=vanishing_point, whether_ceilings=whether_ceilings, whether_floors=whether_floors, whether_walls=whether_walls, \
        whether_boundaries=whether_boundaries, ceiling_places=ceiling_places, floor_places=floor_places)

def visualize_annotation_result(full_save_dir, layout_seg, lines, whether_ceilings, whether_floors, whether_walls, whether_boundaries, ceiling_places, floor_places):
    """Visualize the annotation result

    Args:
        full_save_dir [string]: [the full place to save the picture]
        layout_seg [numpy int array], [H * W]: [the layout segmentation of the picture]
        lines [float array], [(2 * W) * 2]: [the sampled lines, the four instances are top_x, bottom_x]
        whether_ceilings [numpy boolean array], [(2 * W)]: [whether the lines have ceiling]
        whether_floors [numpy boolean array], [(2 * W)]: [whether the lines have floor]
        whether_walls [numpy boolean array], [(2 * W)]: [whether the lines have wall]
        whether_boundaries [numpy boolean array], [(2 * W)]: [whether the lines are wall-wall boundaries]
        ceiling_places [numpy float array], [2 * (2 * W)]: [the ceiling place of each line, (y, x)]
        floor_places [numpy float array], [2 * (2 * W)]: [the floor place of each line, (y, x)]
    """
    H, W = layout_seg.shape
    final_color = layout_seg * 3000

    for i in range(2 * W):
        if whether_ceilings[i] == True:
            y = int(ceiling_places[0, i])
            x = int(ceiling_places[1, i])
            if x >= 0 and x < W and y >= 0 and y < H:
                final_color[y, x] = 65535
        if whether_floors[i] == True:
            y = int(floor_places[0, i])
            x = int(floor_places[1, i])
            if x >= 0 and x < W and y >= 0 and y < H:
                final_color[y, x] = 65535

    for i in range(len(lines)):
        if whether_boundaries[i] != True:
            continue
        line = lines[i]
        top_x, bottom_x = line 
        top_y = 0.0
        bottom_y = H - 1 + 0.0
        dy = 1.0
        dx = (bottom_x - top_x) / bottom_y 
        current_x = top_x 
        current_y = top_y
        for j in range(H):
            axis_x = int(current_x)
            axis_y = int(current_y)
            if axis_x >= 0 and axis_x < W and axis_y >= 0 and axis_y < H and layout_seg[axis_y, axis_x] > 0:
                final_color[axis_y, axis_x] = 65535
            current_y += dy
            current_x += dx 


    final_color = final_color.astype(np.uint16)
    picture = Image.fromarray(final_color)
    picture.save(full_save_dir)
    
         
