import os
import json 
from math import *
import numpy as np
import argparse
import glob
import cv2

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
        labels [numpy int array], [F]: [labels of each face]
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

def save_floorplan(save_place, points, edges, edge_labels, room_range):
    """Save the floorplan of the scene
        P: number of points in the floor plan
        E: number of edges in the floor plan
        
    Args:
        save_place [string]: [the saving place of the floorplan]
        points [numpy float array], [P * 2]: [points in the 2d floor plan]
        edges [numpy int array], [E * 2]: [the id of the start and end points in the 2d floor plan]
        edge_labels [numpy int array], [E]: [the labels of each edges]
        room_range [numpy float array], [2 * 3]: [the lower bound and upper bound of the room, (x, y, z)]
    """
    save_dict = {'points': points.tolist(), 'edges': edges.tolist(), 'labels': edge_labels.tolist(), 'range': room_range.tolist()}
    json_data = json.dumps(save_dict)
    f = open(save_place, 'w')
    f.write(json_data)
    f.close()
    
def load_floorplan(save_place):
    """Load the floorplan of the scene
        P: number of points in the floor plan
        E: number of edges in the floor plan
        
    Args:
        save_place [string]: [the saving place of the floorplan]
        
    Returns:
        points [numpy float array], [P * 2]: [points in the 2d floor plan]
        edges [numpy int array], [E * 2]: [the id of the start and end points in the 2d floor plan]
        edge_labels [numpy int array], [E]: [the labels of each edges]
        room_range [numpy float array], [2 * 3]: [the lower bound and upper bound of the room, (x, y, z)]
    """
    with open(save_place, 'r', encoding='utf8')as fp:
        data = json.load(fp)
    points = data['points']
    edges = data['edges']
    edge_labels = data['labels']
    room_range = data['range']
    points = np.array(points, dtype=np.float32)
    edges = np.array(edges, dtype=np.int32)
    edge_labels = np.array(edge_labels, dtype=np.int32)
    room_range = np.array(room_range, dtype=np.float32)
    return points, edges, edge_labels, room_range

def visualize_floorplan(save_place, points, edges, edge_labels, room_range):
    """Visualize the floorplan
        min:(-2.006484, -3.001466)
        max:(12.124748, 18.227722)
        we set the pictures' range to be (-4, -6) to (16, 24), the picture size is 800 * 1200
    
        P: number of points in the floor plan
        E: number of edges in the floor plan
        
    Args:
        save_place [string]: [the saving place of the floorplan]
        points [numpy float array], [P * 2]: [points in the 2d floor plan]
        edges [numpy int array], [E * 2]: [the id of the start and end points in the 2d floor plan]
        edge_labels [numpy int array], [E]: [the labels of each edges]
        room_range [numpy float array], [2 * 3]: [the lower bound and upper bound of the room, (x, y, z)]
    """
    P = points.shape[0]
    E = edges.shape[0]
    min_x = -4.0 
    min_y = -6.0
    max_x = 16.0
    max_y = 24.0
    size_x = 800
    size_y = 1200
    image = np.zeros((size_y, size_x, 3), np.uint8)
    point_color = (255, 255, 255)
    thickness = 1 
    line_type = 8
    
    for i in range(E):
        start = int(edges[i, 0])
        end = int(edges[i, 1])
        start_x = float(points[start, 0])
        start_y = float(points[start, 1])
        end_x = float(points[end, 0])
        end_y = float(points[end, 1])
        place_start_x = int((start_x - min_x) / (max_x - min_x) * size_x)
        place_start_y = int((start_y - min_y) / (max_y - min_y) * size_y)
        place_end_x = int((end_x - min_x) / (max_x - min_x) * size_x)
        place_end_y = int((end_y - min_y) / (max_y - min_y) * size_y)
        point_start = (place_start_y, place_start_x)
        point_end = (place_end_y, place_end_x)
        cv2.line(image, point_start, point_end, point_color, thickness, line_type)

    cv2.imwrite(save_place, image)

def get_floorplan(vertexs, faces, norms, labels, threshold_norm=0.3):
    """Get the floorplan of the scene
    
        V: number of vertexs of 3D model
        F: number of faces of 3D model
        P: number of points in the floor plan
        E: number of edges in the floor plan
        
    Args:
        vertexs [numpy float array], [V * 3]: [vertexs]
        faces [numpy int array], [F * 3]: [faces]
        norms [numpy float array], [F * 3]: [norms of each face]
        labels [numpy int array], [F]: [labels of each face]
        threshold_norm [float, 0.1 by default]: [the threshold that marks that the plane is a wall]

    Returns:
        points [numpy float array], [P * 2]: [points in the 2d floor plan]
        edges [numpy int array], [E * 2]: [the id of the start and end points in the 2d floor plan]
        edge_labels [numpy int array], [E]: [the labels of each edges]
        room_range [numpy float array], [2 * 3]: [the lower bound and upper bound of the room, (x, y, z)]
    """
    V = vertexs.shape[0]
    F = faces.shape[0]
    wall_points = {}
    
    threshold_bottom = np.mean(vertexs[:, 2])
    
    #get the walls and their points
    for i in range(F):
        label = int(labels[i])
        face = faces[i]
        a = int(face[0])
        b = int(face[1])
        c = int(face[2])
        nz = norms[i, 2]
        if label != 0 and abs(nz) <= threshold_norm:
            if not label in wall_points.keys():
                wall_points[label] = set()
            wall_points[label].add(a)
            wall_points[label].add(b)
            wall_points[label].add(c)

    walls_old = []
    edge_labels = []
    point_pair = {}
    useful_points = set()
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
            return [], [], [], []

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
        max_x = -inf
        max_y = -inf 
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
        edge_labels.append(label)
        useful_points.add(start_id)
        useful_points.add(end_id)
    
    #change the place of points in the floor plan, and get the points and edges
    points = []
    edges = []
    new_index_for_points = {}
    total_down_z = 0.0
    total_up_z = 0.0
    num = 0
    min_x = inf 
    max_x = -inf 
    min_y = inf 
    max_y = -inf
    for down in useful_points:
        up = point_pair[down]
        down_x = float(vertexs[down, 0])
        down_y = float(vertexs[down, 1])
        down_z = float(vertexs[down, 2])
        up_x = float(vertexs[up, 0])
        up_y = float(vertexs[up, 1])
        up_z = float(vertexs[up, 2])
        new_x = (down_x + up_x) / 2.0
        new_y = (down_y + up_y) / 2.0
        total_down_z += down_z
        total_up_z += up_z
        num += 1 
        if new_x < min_x:
            min_x = new_x
        if new_x > max_x:
            max_x = new_x
        if new_y < min_y:
            min_y = new_y
        if new_y > max_y:
            max_y = new_y
        points.append([new_x, new_y])
        new_index_for_points[down] = len(points) - 1
    if num <= 0:
        return [], [], [], []
    bottom_z = total_down_z / num 
    top_z = total_up_z / num
    
    for i in range(len(walls_old)):
        wall_old = walls_old[i]
        start = wall_old[0]
        end = wall_old[1]
        new_start = new_index_for_points[start]
        new_end = new_index_for_points[end]
        edges.append([new_start, new_end])
    
    points = np.array(points, dtype=np.float32)
    edges = np.array(edges, dtype=np.int32)
    edge_labels = np.array(edge_labels, dtype=np.int32)
    room_range = np.array([[min_x, min_y, bottom_z], [max_x, max_y, top_z]], dtype=np.float32)
    
    return points, edges, edge_labels, room_range
    
def main():
    """The main function
    """
    parser = argparse.ArgumentParser(description = '')
    #parser.add_argument('--base_dir', default='/home1/shenguanlin/scannet_planes', type=str)
    parser.add_argument('--base_dir', default='G:\\dataset\\scannet_planes_mine', type=str)

    args = parser.parse_args()
    full_name_list = glob.glob(os.path.join(args.base_dir, '*.ply'))
    scene_id_list = []
    for full_name in full_name_list:
        scene_id = full_name.split(os.sep)[-1][:-4]
        if scene_id[-5:] == '_full':
            continue
        scene_id_list.append(scene_id)
    scene_id_list.sort()
    right_num = 0
    wrong_num = 0

    for scene_id in scene_id_list:
        save_place_json = os.path.join(args.base_dir, scene_id + '_floor_plan.json')
        save_place_picture = os.path.join(args.base_dir, scene_id + '_floor_plan.png')
        vertexs, faces, norms, labels = load_planes(args.base_dir, scene_id)
        points, edges, edge_labels, room_range = get_floorplan(vertexs, faces, norms, labels)
        if len(points) > 0:
            save_floorplan(save_place_json, points, edges, edge_labels, room_range)
            visualize_floorplan(save_place_picture, points, edges, edge_labels, room_range)
            print('Rendered', scene_id)
            right_num += 1
        else:
            print('Error in rendering', scene_id)
            wrong_num += 1
    print(right_num, wrong_num)

if __name__ == "__main__":
    main()