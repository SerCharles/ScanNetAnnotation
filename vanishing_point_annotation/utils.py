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
        if len(the_boundaries) > H / 25:
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

def get_wall_seg(layout_seg, ceiling_id, floor_id):
    """Get the pixels whose backgrounds are walls

    Args:
        layout_seg [numpy int array], [H * W]: [the layout segmentation of the picture]
        ceiling_id [int]: [the id of the plane which is the ceiling]
        floor_id [int]: [the id of the plane which is the floor]
    
    Returns:
        wall_seg [numpy bool array], [H * W]: [the pixels whose backgrounds are walls]
    """
    wall_seg = (layout_seg > 0) & (layout_seg != ceiling_id) & (layout_seg != floor_id)
    return wall_seg


def get_boundaries_per_pixel(layout_seg, vanishing_point, boundary_angles, ceiling_id, floor_id):
    """Get the wall-wall boundaries 
        H: the height of the picture
        W: the width of the picture
        M: the number of boundaries
        
    Args:
        layout_seg [numpy int array], [H * W]: [the layout segmentation of the picture]
        vanishing_point [numpy float array], [2]: [the vanishing point of the picture, (y, x)] 
        boundary_angles [numpy float array], [M]: [the absolute angles of the boundaries]   
        ceiling_id [int]: [the id of the plane which is the ceiling]
        floor_id [int]: [the id of the plane which is the floor]

    Return:
        whether_boundary_per_pixel [numpy bool array], [H * W]: [whether each pixels are boundaries]
    """
    H, W = layout_seg.shape
    whether_boundary_per_pixel = np.zeros((H, W), dtype=np.bool)
    vy = float(vanishing_point[0])
    vx = float(vanishing_point[1])
    for i in range(len(boundary_angles)):
        #get initial place
        if vy > H - 1:
            y0 = H - 1
        else: 
            y0 = 0
        dy0 = abs(vy - y0)
        cos_theta = cos(boundary_angles[i])
        dx0 = cos_theta * dy0 / sqrt(1 - cos_theta ** 2)
        x0 = vx - dx0
        if vy > H - 1:
            y0 = 0
            x0 = x0 - dx0 * (H - 1) / dy0 
        dy = 1.0
        dx = (vx - x0) / (vy - y0)
        y = y0 
        x = x0 

        #iteration
        for j in range(H):
            y_place = int(y)
            x_place = int(x)
            if y_place >= 0 and y_place < H and x_place >= 0 and x_place < W:
                seg = layout_seg[y_place, x_place]
                if seg > 0 and seg != ceiling_id and seg != floor_id:
                    whether_boundary_per_pixel[y_place, x_place] = True 
            y += dy 
            x += dx 
    return whether_boundary_per_pixel

def get_boundary_probability_per_pixel(layout_seg, vanishing_point, boundary_angles, ceiling_id, floor_id, decay_rate=0.96):
    """Get the wall-wall boundary probability
        H: the height of the picture
        W: the width of the picture
        M: the number of boundaries
        
    Args:
        layout_seg [numpy int array], [H * W]: [the layout segmentation of the picture]
        vanishing_point [numpy float array], [2]: [the vanishing point of the picture, (y, x)] 
        boundary_angles [numpy float array], [M]: [the absolute angles of the boundaries]   
        ceiling_id [int]: [the id of the plane which is the ceiling]
        floor_id [int]: [the id of the plane which is the floor]

    Return:
        boundary_probability_per_pixel [numpy float array], [H * W]: [whether each pixels are boundaries]
    """
    H, W = layout_seg.shape
    M = boundary_angles.shape[0]    
    if M <= 0:
        return np.zeros((H, W), dtype=np.float32)
    vy = float(vanishing_point[0])
    vx = float(vanishing_point[1])
    if vy > H - 1:
        dy = vy - H + 1
    else:
        dy = -vy
    log_decay_rate_angle = pi / asin(1 / sqrt(dy ** 2 + 1)) / 180 * log(decay_rate)
    
    
    x, y = np.meshgrid(np.array([ii for ii in range(W)]), np.array([ii for ii in range(H)])) #H * W
    x = x.reshape(H * W)
    y = y.reshape(H * W)
    dx = -(x - vx) #(H * W)
    dy = -(y - vy) #(H * W)
    angles = np.arccos(dx / np.sqrt(dx ** 2 + dy ** 2)) #(H * W)
    angles = angles.reshape(H * W, 1).repeat(M, axis=1) #(H * W) * M
    angles_gt = boundary_angles.reshape(1, M).repeat(H * W, axis=0) #(H * W) * M
    dist_angles = np.abs(angles - angles_gt) #(H * W) * M
    min_angle = np.min(dist_angles, axis=1).reshape(H, W) #H * W
    boundary_probability_per_pixel = np.exp(min_angle * 180 / pi * log_decay_rate_angle)
    mask = (layout_seg > 0) & (layout_seg != ceiling_id) & (layout_seg != floor_id)
    boundary_probability_per_pixel = boundary_probability_per_pixel * mask 
    return boundary_probability_per_pixel
    