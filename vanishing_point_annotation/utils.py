"""The mathematics util functions of vanishing point annotation
"""
import os
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
        vanishing_y [float]: [the x of the vanishing point of the picture]    
        vanishing_x [float]: [the y of the vanishing point of the picture]   
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
    return vanishing_y, vanishing_x


def get_lines(H, W, vanishing_y, vanishing_x):
    """Get the lines from the vanishing point to each top and bottom points 

    Args:
        H [int]: [the height of the picture]
        W [int]: [the width of the picture]
        vanishing_y [float]: [the y of the vanishing point of the picture]    
        vanishing_x [float]: [the x of the vanishing point of the picture] 

    Return:
        lines [float array], [(2 * W) * 2]: [the sampled lines, the four instances are top_x, bottom_x]
    """
    lines = []
    current_y = 0 
    for current_x in range(W):
        dy = vanishing_y - current_y
        dx = vanishing_x - current_x
        dx = dx / dy 
        dy = 1.0
        top_x = current_x + 0.0
        top_y = current_y + 0.0
        bottom_x = top_x + dx * (H - 1)
        bottom_y = H - 1 + 0.0
        line = [top_x, bottom_x]
        lines.append(line)

    current_y = H - 1
    for current_x in range(W):
        dy = vanishing_y - current_y
        dx = vanishing_x - current_x
        dx = dx / dy 
        dy = 1.0
        bottom_x = current_x + 0.0
        bottom_y = current_y + 0.0
        top_x = bottom_x - dx * (H - 1)
        top_y = 0.0
        line = [top_x, bottom_x]
        lines.append(line)

    return lines

def get_ceiling_and_floor(layout_seg, lines, ceiling_id, floor_id):
    """Get the ceiling place and floor place of each line from the vanishing point to the picture, and also whether there are ceiling and floor 
        H: the height of the picture
        W: the width of the picture

    Args:
        layout_seg [numpy int array], [H * W]: [the layout segmentation of the picture]
        lines [float array], [(2 * W) * 2]: [the sampled lines, the four instances are top_x, bottom_x]
        ceiling_id [int]: [the id of the plane which is the ceiling]
        floor_id [int]: [the id of the plane which is the floor]

    Return:
        whether_ceilings [numpy boolean array], [(2 * W)]: [whether the lines have ceiling]
        whether_floors [numpy boolean array], [(2 * W)]: [whether the lines have floor]
        whether_walls [numpy boolean array], [(2 * W)]: [whether the lines have wall]
        ceiling_places [numpy float array], [2 * (2 * W)]: [the ceiling place of each line, (y, x)]
        floor_places [numpy float array], [2 * (2 * W)]: [the floor place of each line, (y, x)]
    """
    H, W =  layout_seg.shape
    whether_ceilings = np.zeros((2 * W), dtype=np.bool)
    ceiling_places = np.zeros((2, 2 * W), dtype=np.float32)
    whether_floors = np.zeros((2 * W), dtype=np.bool)
    floor_places = np.zeros((2, 2 * W), dtype=np.float32)
    whether_walls = np.zeros((2 * W), dtype=np.bool)

    for i in range(len(lines)):
        line = lines[i]
        top_x, bottom_x = line 
        top_y = 0.0
        bottom_y = H - 1 + 0.0
        dy = 1.0
        dx = (bottom_x - top_x) / bottom_y 
        whether_ceiling = False 
        whether_floor = False 
        whether_wall = False
        ceiling_x = top_x
        ceiling_y = top_y 
        floor_x = bottom_x
        floor_y = bottom_y 

        current_x = top_x 
        current_y = top_y
        for j in range(H):
            axis_x = int(current_x)
            axis_y = int(current_y)
            if axis_x < 0 or axis_x >= W or axis_y < 0 or axis_y >= H or layout_seg[axis_y, axis_x] <= 0:
                pass
            elif layout_seg[axis_y, axis_x] == ceiling_id:
                whether_ceiling = True 
                if current_y > ceiling_y: 
                    ceiling_x = current_x
                    ceiling_y = current_y 
            elif layout_seg[axis_y, axis_x] == floor_id:
                whether_floor = True 
                if current_y < floor_y:
                    floor_x = current_x 
                    floor_y = current_y 
            else: 
                whether_wall = True 
            current_x += dx 
            current_y += dy
            
        whether_ceilings[i] = whether_ceiling
        whether_floors[i] = whether_floor
        whether_walls[i] = whether_wall
        ceiling_places[0][i] = ceiling_y
        ceiling_places[1][i] = ceiling_x 
        floor_places[0][i] = floor_y
        floor_places[1][i] = floor_x
    
    return whether_ceilings, whether_floors, whether_walls, ceiling_places, floor_places



def get_wall_boundaries(layout_seg, lines, ceiling_id, floor_id):
    """Get the wall-wall boundaries 

    Args:
        layout_seg [numpy int array], [H * W]: [the layout segmentation of the picture]
        lines [float array], [(2 * W) * 2]: [the sampled lines, the four instances are top_x, bottom_x]
        ceiling_id [int]: [the id of the plane which is the ceiling]
        floor_id [int]: [the id of the plane which is the floor]

    Return:
        whether_boundaries [numpy boolean array], [(2 * W)]: [whether the lines are wall-wall boundaries]
    """
    H, W =  layout_seg.shape
    whether_boundaries = np.zeros((2 * W), dtype=np.bool)

    max_id = np.max(layout_seg)
    boundaries = []
    for i in range((max_id + 1) ** 2):
        boundaries.append([])

    for y in range(H):
        for x in range(W - 1):
            left_seg = 0
            right_seg = 0
            if layout_seg[y, x] > 0 and layout_seg[y, x] != ceiling_id and layout_seg[y, x] != floor_id:
                left_seg = layout_seg[y, x]
            if layout_seg[y, x + 1] > 0 and layout_seg[y, x + 1] != ceiling_id and layout_seg[y, x + 1] != floor_id:
                right_seg = layout_seg[y, x + 1]
            if left_seg > 0 and right_seg > 0 and left_seg != right_seg:
                seg = left_seg * (max_id + 1) + right_seg
                boundaries[seg].append([y, x])
    

    for i in range(len(boundaries)):
        boundary = boundaries[i]
        if len(boundary) > (H / 10):
            np_boundary = np.array(boundary)
            y = np_boundary[:, 0].reshape(-1, 1)
            x = np_boundary[:, 1]
            reg = LinearRegression().fit(y, x)
            k = reg.coef_[0]
            b = reg.intercept_ #ky - x + b = 0

            min_distance = sqrt(W ** 2 + H ** 2)
            best_id = -1

            for j in range(W):
                line = lines[j]
                top_x, bottom_x = line 
                dist_top = abs((k * 0.0 - top_x + b) / sqrt(k ** 2 + 1))
                dist_bottom = abs((k * (H - 1) - bottom_x + b) / sqrt(k ** 2 + 1))
                dist = (dist_top + dist_bottom) / 2
                if dist < min_distance:
                    min_distance = dist 
                    best_id = j 
            if best_id >= 0 and best_id < W:
                whether_boundaries[best_id] = True 
            

            min_distance = sqrt(W ** 2 + H ** 2)
            best_id = -1
            for j in range(W):
                line = lines[j + W]
                top_x, bottom_x = line 
                dist_top = abs((k * 0.0 - top_x + b) / sqrt(k ** 2 + 1))
                dist_bottom = abs((k * (H - 1) - bottom_x + b) / sqrt(k ** 2 + 1))
                dist = (dist_top + dist_bottom) / 2
                if dist < min_distance:
                    min_distance = dist 
                    best_id = j 
            if best_id >= 0 and best_id < W:
                whether_boundaries[best_id + W] = True   

    return whether_boundaries

