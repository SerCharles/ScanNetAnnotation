"""Get the background mesh based on the json file of planes of ScanNet-Planes dataset
"""
import numpy as np
import os
from math import *
import json
import argparse
import glob

def write_ply_file(filename, points, faces):
    """Write a ply file with only points and faces
        V: the number of vertexs
        F: the number of faces

    Args:
        filename [string]: [the full path of the ply file]
        points [numpy float array], [V * 3]: [the 3D point lists]
        faces [numpy int array], [F * 3]: [the triangular face lists]
    """

    with open(filename, 'w') as f:
        header = "ply\n" + \
                "format ascii 1.0\n" + \
                "element vertex " + \
                str(len(points)) + '\n' + \
                "property float x\n" + \
                "property float y\n" + \
                "property float z\n" + \
                "element face " + \
                str(len(faces)) + "\n" + \
                "property list uchar int vertex_index\n" + \
                "end_header\n"
        f.write(header)
        for point in points:
            f.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
            continue
        for face in faces:
            f.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')

            continue     
        f.close()
        pass
    return

def save_plane_info(filename, face_labels, face_norms, ceiling_id, floor_id):
    """Save the plane info in the json file
        V: the number of vertexs
        F: the number of faces

    Args:
        filename [string]: [the saving full path]
        face_labels [int array], [F]: [the plane label of each triangular mesh face]
        face_norms [float array], [F * 3]: [the normal of each triangular mesh face]
        ceiling_id [int]: [the id of the plane which is the ceiling]
        floor_id [int]: [the id of the plane which is the floor]
    """
    save_dict = {'labels': face_labels, 'norms': face_norms, 'ceiling_id': ceiling_id, 'floor_id': floor_id}
    json_data = json.dumps(save_dict)
    f = open(filename, 'w')
    f.write(json_data)
    f.close()

def calculate_normal(point_lists):
    """Calculate the normal of the plane based on the points
        M: the number of points

    Args:
        point_lists [float array], [M * 3]: [the point lists]

    Returns:
        normal [float array], [3]: [the normal info, which is normalized]
    """
    ones = np.ones(len(point_lists))
    point_array = np.array(point_lists, dtype = np.float32)
    normal_numpy = np.matmul(np.matmul(np.linalg.inv(np.matmul(point_array.T, point_array)), point_array.T), ones)
    nx = normal_numpy[0]
    ny = normal_numpy[1]
    nz = normal_numpy[2]
    length = sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx = nx / length 
    ny = ny / length 
    nz = nz / length
    normal = [nx, ny, nz]
    return normal

def calculate_center(point_lists):
    """Calculate the center of the plane based on the points
        M: the number of points

    Args:
        point_lists [float array], [M * 3]: [the point lists]

    Returns:
        center [float array], [3]: [the center info]
    """
    point_array = np.array(point_lists, dtype = np.float32) #M * 3
    center = np.mean(point_array, axis=0).tolist()
    return center

def process_one_scene(base_dir_source, base_dir_target, scene_id):
    """Process one scene of ScanNet-Planes dataset, process one scene at once

    Args:
        base_dir_source [string]: [the base directory of ScanNet-Planes dataset]
        base_dir_target [string]: [the base directory of our saving place]
        scene_id [string]: [the scene id]
    """
    full_name_info = os.path.join(base_dir_source, scene_id + '.json')
    save_name_ply = os.path.join(base_dir_target, scene_id + '.ply')
    save_name_info = os.path.join(base_dir_target, scene_id + '.json')
    with open(full_name_info, 'r', encoding='utf8')as fp:
        data = json.load(fp)
    vertexs = [[0.0, 0.0, 0.0]] + data['verts']
    planes = data['quads']
    faces = [[0, 0, 0]]
    face_labels = [0]
    face_norms = [[0.0, 0.0, 0.0]]

    label_ceiling = -1
    label_floor = -1
    
    for i in range(len(vertexs)):
        vertex = vertexs[i]
        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        vertexs[i][2] = y 
        vertexs[i][1] = -z
    
    for i in range(len(planes)):
        plane = planes[i]
        point_list = []
        for j in range(len(plane)):
            point_list.append(vertexs[plane[j] + 1])
        normal = calculate_normal(point_list)
        center = calculate_center(point_list)
        
        if abs(normal[2]) >= 0.9:
            if abs(center[2]) <= 0.2:
                label_floor = i + 1
            else: 
                label_ceiling = i + 1
        
        for j in range(len(plane) - 2):
            new_face = [plane[0] + 1, plane[j + 1] + 1, plane[j + 2] + 1]
            faces.append(new_face)
            face_labels.append(i + 1)
            face_norms.append(normal)

    write_ply_file(save_name_ply, vertexs, faces)
    save_plane_info(save_name_info, face_labels, face_norms, label_ceiling, label_floor)


def main():
    """[the main function of ScanNet-Planes dataset processing]
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir_source', default='/home1/sgl/scannet_planes', type=str)
    parser.add_argument('--base_dir_target', default='/home1/sgl/scannet_planes_mine', type=str)
    args = parser.parse_args()
    full_name_list = glob.glob(os.path.join(args.base_dir_source, '*.ply'))
    for full_name in full_name_list:
        scene_id = full_name.split(os.sep)[-1][:-4]
        print('Rendering', scene_id)
        process_one_scene(args.base_dir_source, args.base_dir_target, scene_id)



if __name__ == "__main__":
    main()