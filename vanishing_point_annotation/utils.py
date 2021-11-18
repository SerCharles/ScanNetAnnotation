"""The mathematics util functions of vanishing point annotation
"""
from math import *
import numpy as np
from sklearn.linear_model import LinearRegression


def get_vanishing_point(H, W, intrinsic, extrinsic):
    """Get the vanishing point of direction(0, 0, 1) in the picture

    Args:
        H [int]: [the height of the picture]
        W [int]: [the width of the picture]
        intrinsic [numpy float array], [3 * 3]: [the intrinsic of the picture]
        extrinsic [numpy float array], [4 * 4]: [the extrinsic of the picture]
        
    Returns:
        vanishing_point [numpy float array], [2]: [the vanishing point of the picture, (y, x)]    
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
    vanishing_point = np.array([vanishing_y, vanishing_x], dtype=np.float32)
    return vanishing_point


def get_wall_boundaries(layout_seg, vanishing_point, ceiling_id, floor_id):
    """Get the wall-wall boundaries 
        H: the height of the picture
        W: the width of the picture
        M: the number of boundaries
        
    Args:
        layout_seg [numpy int array], [H * W]: [the layout segmentation of the picture]
        vanishing_point [numpy float array], [2]: [the vanishing point of the picture, (y, x)]    
        ceiling_id [int]: [the id of the plane which is the ceiling]
        floor_id [int]: [the id of the plane which is the floor]

    Return:
        boundary_angles [numpy float array], [M]: [the absolute angles of the boundaries]
        boundary_segs [numpy int array], [M * 2]: [the left and right seg numbers of the boundaries]
    """
    H, W = layout_seg.shape
    layout_seg_left = layout_seg[:, 0:W - 1]
    layout_seg_right = layout_seg[:, 1:W]
    zeros = np.zeros((H, 1), dtype=np.bool)
    
    whether_boundary = (layout_seg_left > 0) & (layout_seg_left != ceiling_id) & (layout_seg_left != floor_id) & \
        (layout_seg_right > 0) & (layout_seg_right != ceiling_id) & (layout_seg_right != floor_id) & (layout_seg_left != layout_seg_right)
    whether_boundary = np.concatenate((whether_boundary, zeros), axis=1)

    x_indices, y_indices = np.meshgrid(np.array([ii for ii in range(W)]), np.array([ii for ii in range(H)]))

    y_indices = y_indices.reshape(H * W)
    x_indices = x_indices.reshape(H * W)
    whether_boundary = whether_boundary.reshape(H * W)
    boundary_y = y_indices[whether_boundary]
    boundary_x = x_indices[whether_boundary]
    num_boundary = boundary_y.shape[0]
    
    boundary_angles = []
    boundary_segs = []
    max_id = np.max(layout_seg)
    boundaries = []
    for i in range((max_id + 1) ** 2):
        boundaries.append([])
    
    for i in range(num_boundary):
        by = int(boundary_y[i])
        bx = int(boundary_x[i])
        left_seg = layout_seg[by, bx]
        right_seg = layout_seg[by, bx + 1]
        seg = left_seg * (max_id + 1) + right_seg
        boundaries[seg].append([by, bx])
    
    for i in range(len(boundaries)):
        the_boundaries = boundaries[i]
        if len(the_boundaries) > 0:
            angles_total = 0.0
            for point in the_boundaries:
                y, x = point 
                vy = float(vanishing_point[0])
                vx = float(vanishing_point[1])
                dx = vx - x 
                dy = vy - y
                angle = acos(dx / sqrt(dx ** 2 + dy ** 2))
                angles_total += angle
            avg_angle = angles_total / len(the_boundaries)
            left_seg = i // (max_id + 1)
            right_seg = i % (max_id + 1)
            boundary_angles.append(avg_angle)
            boundary_segs.append([left_seg, right_seg])
    
    boundary_angles = np.array(boundary_angles, dtype=np.float32)
    boundary_segs = np.array(boundary_segs, dtype=np.int32)
    return boundary_angles, boundary_segs


def get_whether_boundaries(H, W, vanishing_point, boundary_angles):
    """Get whether the lines are boundaries

    Args:
        H [int]: [the height of the picture]
        W [int]: [the width of the picture]
        vanishing_point [numpy float array], [2]: [the vanishing point of the picture, (y, x)]    
        boundary_angles [numpy float array], [M]: [the absolute angles of the boundaries]
        
    Returns:
        whether_boundaries [numpy bool array], [W]: [whether the lines are boundaries]
    """
    whether_boundaries = np.zeros(W, dtype=np.bool)
    M = boundary_angles.shape[0]
    vy = float(vanishing_point[0])
    vx = float(vanishing_point[1])
    if vy > H - 1:
        y = H - 1
    else: 
        y = 0
    dy = abs(vy - y)
    for i in range(M):
        cos_theta = cos(boundary_angles[i])
        dx = cos_theta * dy / sqrt(1 - cos_theta ** 2)
        x = int(vx - dx) 
        if x <= W - 1 and x >= 0:
            whether_boundaries[x] = True 
    return whether_boundaries
