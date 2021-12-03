import os
import json 
from math import *
import numpy as np

def load_planes(base_dir, scene_id):
    """Load the ScanNet-Planes dataset
        V: number of vertexs
        F: number of faces

    Args:
        base_dir [string]: [the base directory of ScanNet-Planes dataset]
        scene_id [string]: [the scene id]
    
    Returns:
        vertexs [numpy float array], [V * 3]: [vertexs]
        faces [numpy int array], [F * 3]: [faces]
        norms [numpy float array], [F * 3]: [norms of each face]
        labels [numpy int array], [F * 1]: [labels of each face]
    """

    full_name_ply = os.path.join(base_dir, scene_id + '.ply')
    full_name_info = os.path.join(base_dir, scene_id + '.json')
    vertexs = []
    faces = []
    f = open(full_name_ply, 'r')
    lines = f.read().split('\n')
    f.close()
    vertex_num = int(lines[2].split()[-1])
    face_num = int(lines[6].split()[-1])
    for i in range(vertex_num):
        line = lines[i + 9]
        words = line.split()
        x = float(words[0])
        y = float(words[1])
        z = float(words[2])
        vertex = [x, y, z]
        vertexs.append(vertex)
    for i in range(face_num):
        line = lines[i + vertex_num + 9]
        words = line.split()
        a = int(words[1])
        b = int(words[2])
        c = int(words[3])
        face = [a, b, c]
        faces.append(face)
    with open(full_name_info, 'r', encoding='utf8')as fp:
        data = json.load(fp)
    norms = data['norms']
    labels = data['labels']

    vertexs = np.array(vertexs, dtype='float32')
    faces = np.array(faces, dtype='int32')
    norms = np.nan_to_num(np.array(norms, dtype='float32'))
    labels = np.array(labels, dtype='int32')
    return vertexs, faces, norms, labels



def get_floorplan(vertexs, faces, norms, labels, threshold_norm=0.1, threshold_bottom=0.2):
    """Get the floorplan of the scene
        V: number of vertexs of 3D model
        F: number of faces of 3D model
        P: number of points in the floor plan
        E: number of edges in the floor plan
        
    Args:
        vertexs [numpy float array], [V * 3]: [vertexs]
        faces [numpy int array], [F * 3]: [faces]
        norms [numpy float array], [F * 3]: [norms of each face]
        labels [numpy int array], [F * 1]: [labels of each face]
        threshold_norm [float, 0.1 by default]: [the threshold that marks that the plane is a wall]
        threshold_bottom [float, 0.2 by default]: [the threshold that marks that the point is at the bottom]

    Returns:
        points [numpy float array], [P * 2]: [points in the 2d floor plan]
        edges [numpy int array], [E * 2]: [the id of the start and end points in the 2d floor plan]
        edge_labels [numpy int array], [E * 1]: [the labels of each edges]
    """
    V = vertexs.shape[0]
    F = faces.shape[0]
    wall_points = {}
    
    #get the walls and their points
    for i in range(F):
        label = int(labels[i, 0])
        face = faces[i]
        a = int(face[0])
        b = int(face[1])
        c = int(face[2])
        nz = norms[i, 2]
        if abs(nz) <= threshold_norm:
            if not label in wall_points.keys():
                wall_points[label] = {}
            wall_points[label].add(a)
            wall_points[label].add(b)
            wall_points[label].add(c)

    walls_old = []
    edge_labels = []
    point_pair = {}
    useful_points = {}
    for label in wall_points.keys():
        points = wall_points[label]
        up_list = []
        down_list = []
        for point in points:
            z = vertexs[point, 2]
            if z < threshold_bottom:
                down_list.append(point)
            else:
                up_list.append(point)
        if len(up_list) != len(down_list):
            print('Error!')
            return

        #allocate the point pairs
        for down in down_list:
            if down in point_pair.keys():
                continue
            x = vertexs[down, 0]
            y = vertexs[down, 1]
            best_id = -1 
            best_dist = inf 
            for up in up_list:
                up_x = vertexs[up, 0]
                up_y = vertexs[up, 1]
                dist = sqrt((x - up_x) ** 2 + (y - up_y) ** 2)
                if dist < best_dist:
                    best_dist = dist 
                    best_id = up 
            point_pair[down] = best_id
            
        #get the start and end bottom points of the wall
        start_id = -1 
        end_id = -1
        min_x = inf 
        min_y = inf 
        max_x = inf 
        max_y = inf 
        argmin_x = -1 
        argmin_y = -1 
        argmax_x = -1
        argmax_y = -1
        for down in down_list:
            x = vertexs[down, 0]
            y = vertexs[down, 1]
            if x < min_x:
                min_x = x 
                argmin_x = down 
            if x > max_x:
                max_x = x 
                argmax_x = down 
            if y < min_y:
                min_y = y 
                argmin_y = down 
            if y > max_y:
                max_y = y
                argmax_y = down 
        dx = abs(max_x - min_x)
        dy = abs(max_y - min_y)
        if dy > dx: 
            start_id = argmin_y
            end_id = argmax_y
        else:
            start_id = argmin_x
            end_id = argmax_x
        walls_old.append([start_id, end_id])
        edge_labels.append([int(label)])
        useful_points.add(start_id)
        useful_points.add(end_id)
    
    #change the place of points in the floor plan, and get the points and edges
    points = []
    edges = []
    new_index_for_points = {}
    for down in useful_points:
        up = point_pair[point]
        down_x = float(vertexs[down, 0])
        down_y = float(vertexs[down, 1])
        up_x = float(vertexs[up, 0])
        up_y = float(vertexs[up, 1])
        new_x = (down_x + up_x) / 2.0
        new_y = (down_y + up_y) / 2.0 
        points.append([new_x, new_y])
        new_index_for_points[down] = len(points) - 1
    
    for i in range(walls_old):
        wall_old = walls_old[i]
        start = wall_old[0]
        end = wall_old[1]
        new_start = new_index_for_points[start]
        new_end = new_index_for_points[end]
        edges.append([new_start, new_end])
    
    points = np.array(points, dtype=np.float32)
    edges = np.array(edges, dtype=np.int32)
    edge_labels = np.array(edge_labels, dtype=np.int32)
    return points, edges, edge_labels
    
    
        
    