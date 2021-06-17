'''
Generate the ply mesh file and relative info based on the json data
'''
import numpy as np
import os
from math import *
import json
import argparse
import glob

def write_ply_file(filename, points, faces):
    """[write a ply file]

    Args:
        filename ([str]): [the name of the ply file]
        points ([float array]): [the 3D point lists]
        faces ([int array]): [the triangular face lists]
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

def save_plane_info(filename, face_labels, face_norms):
    """[save the plane info in the json file]

    Args:
        filename ([str]): [the saving final place]
        face_labels ([int array]): [the plane label of each triangular mesh face]
        face_norms ([type]): [the normal of each triangular mesh face]
    """
    save_dict = {'labels': face_labels, 'norms': face_norms}
    json_data = json.dumps(save_dict)
    f = open(filename, 'w')
    f.write(json_data)
    f.close()

def calculate_normal(point_lists):
    """[calculate the normal of the plane based on the points]

    Args:
        point_lists ([float array]): [the point lists]

    Returns:
        [float array]: [the normal info, which is normalized]
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

def modify_one_scene(base_dir_source, base_dir_target, scene_id):
    """[modify one scene of ScanNet-Planes dataset]

    Args:
        base_dir_source ([str]): [the base directory of ScanNet-Planes dataset]
        base_dir_target ([str]): [the base directory of our saving place]
        scene_id ([str]): [the scene id]
    """
    full_name_info = os.path.join(base_dir_source, scene_id + '.json')
    save_name_ply = os.path.join(base_dir_target, scene_id + '.ply')
    save_name_info = os.path.join(base_dir_target, scene_id + '.json')
    with open(full_name_info, 'r', encoding = 'utf8')as fp:
        data = json.load(fp)
    vertexs = [[0.0, 0.0, 0.0]] + data['verts']
    planes = data['quads']
    faces = [[0, 0, 0]]
    face_labels = [0]
    face_norms = [[0.0, 0.0, 0.0]]

    '''
    for i in range(len(vertexs)):
        vertex = vertexs[i]
        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        vertexs[i][2] = y 
        vertexs[i][1] = -z
    '''
    for i in range(len(planes)):
        plane = planes[i]
        point_list = []
        for j in range(len(plane)):
            point_list.append(vertexs[plane[j] + 1])
        normal = calculate_normal(point_list)
        for j in range(len(plane) - 2):
            new_face = [plane[0] + 1, plane[j + 1] + 1, plane[j + 2] + 1]
            faces.append(new_face)
            face_labels.append(i + 1)
            face_norms.append(normal)


    write_ply_file(save_name_ply, vertexs, faces)
    save_plane_info(save_name_info, face_labels, face_norms)


def main():
    """[the main function of ScanNet-Planes dataset modification]
    """
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--base_dir_source', default = 'E:\\dataset\\scannet_planes', type = str)
    parser.add_argument('--base_dir_target', default = 'E:\\dataset\\scannet_planes_mine', type = str)
    args = parser.parse_args()
    full_name_list = glob.glob(os.path.join(args.base_dir_source, '*.ply'))
    for full_name in full_name_list:
        scene_id = full_name.split(os.sep)[-1][:-4]
        print('Rendering', scene_id)
        modify_one_scene(args.base_dir_source, args.base_dir_target, scene_id)



if __name__ == "__main__":
    main()