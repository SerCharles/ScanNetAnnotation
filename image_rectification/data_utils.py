"""The utility functions used in data loading
"""


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


def save_extrinsic(file_name, pose):
    """save an extrinsic file of ScanNet

    Args:
        file_name [str]: [the name of the extrinsic file]
        pose [numpy float array], [4 * 4]: [the extrinsic numpy array]
    """
    f = open(file_name, 'w')
    
    for i in range(4):
        f.write(str(pose[i][0]) + ' ' + str(pose[i][1]) + ' ' + str(pose[i][2]) + ' ' + str(pose[i][3]))
        if i != 3:
            f.write('\n')
    f.close()


def get_mask(depth, normal):
    """Get the mask of one picture, masking out those place with useless data
        H: the height of the picture
        W: the width of the picture
        
    Args:
        depth [numpy float array], [1 * H * W]: [depth map]
        normal [numpy float array], [3 * H * W]: [normal map]

    Returns:
        mask [numpy boolean array], [1 * H * W]: [the mask of the data, 1 means ok, 0 means useless]
    """
    mask_depth = depth > 0 
    sqrt_normal = normal[0:1, :, :] ** 2 + normal[1:2, :, :] ** 2 + normal[2:3, :, :] ** 2
    mask_normal = ~(sqrt_normal < 1e-8)
    mask = mask_depth & mask_normal
    return mask