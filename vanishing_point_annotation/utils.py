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
                
def get_relative_angles(H, W, vanishing_point, absolute_angles):
    """Switch the absolute angles to the relative angles
        H: the height of the picture
        W: the width of the picture
        M: the number of boundaries
        
    Args:
        H [int]: [the height of the picture]
        W [int]: [the width of the picture]
        vanishing_point [numpy float array], [2]: [the vanishing point of the picture, (y, x)]    
        absolute_angles [numpy float array], [M]: [the absolute angles of the boundaries]
    
    Returns:
        relative_angles [numpy float array], [M]: [the relative angles of the boundaries]
    """
    vy = float(vanishing_point[0])
    vx = float(vanishing_point[1])
    max_dx = vx - W + 1 
    min_dx = vx
    if vy > H - 1:
        dy = vy - H + 1 
    else: 
        dy = vy
    min_angle = acos(min_dx / sqrt(min_dx ** 2 + dy ** 2))
    max_angle = acos(max_dx / sqrt(max_dx ** 2 + dy ** 2))
    relative_angles = (absolute_angles - min_angle) / (max_angle - min_angle)
    return relative_angles


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
    relative_angles = get_relative_angles(H, W, vanishing_point, boundary_angles)
    boundary_places = np.clip((relative_angles * W).astype(np.int32), 0, W - 1)
    whether_boundaries = np.zeros((W), dtype=np.bool)
    whether_boundaries[boundary_places] = True 
    return whether_boundaries

