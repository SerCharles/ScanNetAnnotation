"""Given the mesh of the background of the modified ScanNet-Planes dataset, generate a dense pointcloud of it, used as baseline to be compared against the result of ours
"""
import numpy as np
import os
from plyfile import *
from math import *
import argparse
import glob

def load_ply(model_path):
    """load a ply file with only point and face
        V: the number of vertexs
        F: the number of faces

    Args:
        model_path [strimng]: [the place of ply file]

    Returns:
        vertexs [numpy float array], [V * 3]: [vertexs]
        faces [numpy int array], [F * 3]: [faces]
    """
    plydata = PlyData.read(model_path)
    my_vertexs = []
    my_faces = []

    vertexs = plydata['vertex']
    faces = plydata['face']
    for i in range(vertexs.count):
        x = float(vertexs[i][0])
        y = float(vertexs[i][1])
        z = float(vertexs[i][2])
        my_vertexs.append([x, y, z])

    for i in range(faces.count):
        a = int(faces[i][0][0])
        b = int(faces[i][0][1])
        c = int(faces[i][0][2])
        my_faces.append([a, b, c])
    
    faces = np.array(my_faces, dtype='int32')
    vertexs = np.array(my_vertexs, dtype='float32')
    return vertexs, faces


def write_pointcloud(filename, points):
    """Write a pointcloud file with only x, y, z
        V: the number of vertexs
        F: the number of faces

    Args:
        filename [string]: [the name of the file to be saved]
        points [numpy float array], [V * 3]: [the 3D point lists of the pointcloud to be saved]
    """

    with open(filename, 'w') as f:
        header = "ply\n" + \
                "format ascii 1.0\n" + \
                "element vertex " + \
                str(len(points)) + '\n' + \
                "property float x\n" + \
                "property float y\n" + \
                "property float z\n" + \
                "end_header\n"
        f.write(header)
        for point in points:
            f.write(str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
            continue   
        f.close()
        pass
    return

def get_dist(a, b):
    """Get the distance between two points
        V: the number of vertexs
        F: the number of faces

    Args:
        a [numpy float array], [3]: [the place of point a]
        b [numpy float array], [3]: [the place of point b]

    Returns:
        dist [float]: [distance]
    """
    dist = sqrt(float(np.dot(b - a, b - a)))
    return dist

def get_area(a, b, c):
    """Given the coordinate of the three points, get the area of the triangle
        V: the number of vertexs
        F: the number of faces

    Args:
        a [numpy float array], [3]: [the place of point a]
        b [numpy float array], [3]: [the place of point b]
        c [numpy float array], [3]: [the place of point c]

    Returns:
        area [float]: [the area of the triangle]
    """
    dist_a = get_dist(a, b)
    dist_b = get_dist(b, c)
    dist_c = get_dist(c, a)
    try:
        area = sqrt((dist_a + dist_b + dist_c) * (dist_a + dist_b - dist_c) * (dist_a - dist_b + dist_c) * (-dist_a + dist_b + dist_c)) / 4
    except: 
        area = 0
    return area

def get_average_dist_and_area(full_file_name):
    """Given a scene mesh, get the average dist between two adjacent points and the average area of the faces
        V: the number of vertexs
        F: the number of faces

    Args:
        full_file_name [string]: [the full path of original mesh]

    Returns:
        avg_dist [float]: [the average dist of between two adjacent points in the original mesh]
        avg_area [float]: [the average area of the triangles in the original mesh]
    """
    plydata = PlyData.read(full_file_name)
    my_vertexs = []
    total_area = 0.0 
    total_num_area = 0
    total_dist = 0.0 
    total_num_dist = 0

    vertexs = plydata['vertex']
    faces = plydata['face']
    for i in range(vertexs.count):
        x = float(vertexs[i][0])
        y = float(vertexs[i][1])
        z = float(vertexs[i][2])
        my_vertexs.append([x, y, z])
    my_vertexs = np.array(my_vertexs, dtype='float32')
    for i in range(faces.count):
        a = int(faces[i][0][0])
        b = int(faces[i][0][1])
        c = int(faces[i][0][2])
        total_dist += get_dist(my_vertexs[a], my_vertexs[b])
        total_dist += get_dist(my_vertexs[b], my_vertexs[c])
        total_dist += get_dist(my_vertexs[c], my_vertexs[a])
        total_num_dist += 3

        total_area += get_area(my_vertexs[a], my_vertexs[b], my_vertexs[c])
        total_num_area += 1

    avg_dist = total_dist / total_num_dist
    avg_area = total_area / total_num_area 
    return avg_dist, avg_area

def get_point_pairs(faces):
    """Get the pairs of adjacent points in a mesh
        V: the number of vertexs
        F: the number of faces

    Args:
        faces [numpy int array], [F * 3]: [the faces of the mesh]
    Returns:
        pairs [set]: [the set of point pairs]
    """
    pairs = set()
    for i in range(len(faces)):
        a = int(faces[i][0])
        b = int(faces[i][1])
        c = int(faces[i][2])

        ab = (min(a, b), max(a, b))
        bc = (min(b, c), max(b, c))
        ac = (min(a, c), max(a, c))
        pairs.add(ab)
        pairs.add(bc)
        pairs.add(ac)
    return pairs

def split_line(a, b, avg_dist):
    """Split a long line into several points
        V: the number of vertexs
        F: the number of faces
        M: the number of new points you split

    Args:
        a [numpy float array], [3]: [the place of point a]
        b [numpy float array], [3]: [the place of point b]
        avg_dist [float]: [the average dist of lines, the function aims at spliting line ab to small sub lines whose lengths are close to it]

    Returns:
        new_points: [numpy float array], [M * 3]: [the place of new points]
    """
    new_points = [] 
    dist = get_dist(a, b)
    num = int(dist / avg_dist)
    if num <= 1:
        return []

    for i in range(1, num):
        j = num - i 
        new_point = (a * i + b * j) / num 
        new_points.append(new_point)
    return new_points

def split_area(a, b, c, avg_area):
    """Split the interior of a big triangle into several points
        V: the number of vertexs
        F: the number of faces
        M: the number of new points you split

    Args:
        a [numpy float array], [3]: [the place of point a]
        b [numpy float array], [3]: [the place of point b]
        c [numpy float array], [3]: [the place of point c]
        avg_area [float]: [the average dist of areas, the function aims at spliting triangle abc to small sub triangles whose areas are close to it]

    Returns:
        new_points: [numpy float array], [M * 3]: [the place of new points]
    """
    new_points = [] 
    area = get_area(a, b, c)
    num = int(sqrt(area / avg_area))
    if num <= 2:
        return []

    for i in range(1, num - 1):
        for j in range(1, num - i):
            k = num - i - j 
            new_point = (a * i + b * j + c * k) / num
            new_points.append(new_point) 
    return new_points

def get_background_pointcloud(base_dir_scannet, base_dir_plane, scene_id):
    """Get the dense background pointcloud based on the original mesh, process only one scene

    Args:
        base_dir_scannet [string]: [the base directory of ScanNet dataset]
        base_dir_plane [string]: [the base directory of ScanNet-Planes dataset]
        scene_id [string]: [the scene id]
    """
    scannet_name_ply = os.path.join(base_dir_scannet, scene_id, 'ply', scene_id + '_vh_clean_2.ply')
    plane_name_ply = os.path.join(base_dir_plane, scene_id + '.ply')
    save_name_ply = os.path.join(base_dir_plane, scene_id + '_full.ply')

    avg_dist, avg_area = get_average_dist_and_area(scannet_name_ply)
    vertexs, faces = load_ply(plane_name_ply)
    lines = get_point_pairs(faces)

    new_points = []

    for line in lines: 
        a, b = line 
        point_a = vertexs[a]
        point_b = vertexs[b]
        points = split_line(point_a, point_b, avg_dist)
        new_points += points 

    for face in faces: 
        a, b, c = face 
        point_a = vertexs[a]
        point_b = vertexs[b]
        point_c = vertexs[c]
        points = split_area(point_a, point_b, point_c, avg_area)
        new_points += points 
    
    new_points = np.array(new_points, dtype='float32')
    result_vertexs = np.concatenate((vertexs, new_points), axis = 0)
    
    write_pointcloud(save_name_ply, result_vertexs)
    print('written', save_name_ply)

def main():
    """The main function
    """
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--base_dir_scannet', default='/home1/sgl/scannet_mine', type=str)
    parser.add_argument('--base_dir_plane', default='/home1/sgl/scannet_planes', type=str)
    args = parser.parse_args()
    full_name_list = glob.glob(os.path.join(args.base_dir_plane, '*.ply'))
    for full_name in full_name_list:
        scene_id = full_name.split(os.sep)[-1][:-4]
        if scene_id[-5:] == '_full':
            continue
        print('Rendering', scene_id)
        get_background_pointcloud(args.base_dir_scannet, args.base_dir_plane, scene_id)

if __name__ == "__main__":
    main()
