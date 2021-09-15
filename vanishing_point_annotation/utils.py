"""The mathematics util functions of vanishing point annotation
"""
import os
import numpy as np



def get_vanishing_point(H, W, intrinsic, extrinsic):
    """Get the vanishing point of direction(0, 0, 1) in the picture

    Args:
        H [int]: [the height of the picture]
        W [int]: [the width of the picture]
        intrinsic [numpy float array], [3 * 3]: [the intrinsic of the picture]
        extrinsic [numpy float array], [4 * 4]: [the extrinsic of the picture]
        
    Returns:
        vanishing_y [float]: [the y of the vanishing point of the picture]    
        vanishing_x [float]: [the x of the vanishing point of the picture]   
    """
    rotation = extrinsic[0:3, 0:3] #3 * 3
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[2, 0]
    cy = intrinsic[2, 1]
    d_camera = np.array([0, 0, 1]).astype(np.float32)
    d_world = np.matmul(np.linalg.inv(rotation), d_camera)
    dx = d_world[0]
    dy = d_world[1]
    dz = d_world[2]
    vanishing_x = fx * dx / dz + cx 
    vanishing_y = fy * dy / dz + cy 
    return vanishing_x, vanishing_y


