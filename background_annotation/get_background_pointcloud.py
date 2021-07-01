'''
Get the full pointcloud of the background
'''
import numpy as np
import os
from plyfile import *
from math import *
import argparse
import glob

def load_ply(model_path):
    """[load a ply file]
    Args:
        model_path ([str]): [the place of ply file]

    Returns:
        V [numpy float array]: [vertexs]
        F [numpy int array]: [faces]
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
    
    F = np.array(my_faces, dtype = 'int32')
    V = np.array(my_vertexs, dtype = 'float32')
    return V, F

def write_pointcloud_file(filename, points):
    """[write a pointcloud file]

    Args:
        filename ([str]): [the name of the ply file]
        points ([float array]): [the 3D point lists]
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
    """[get the distance between two points]

    Args:
        a ([numpy float array]): [the place of point a]
        b ([numpy float array]): [the place of point b]

    Returns:
        dist [float]: [distance]
    """
    dist = sqrt(float(np.dot(b - a, b - a)))
    return dist

def get_area(a, b, c):
    """[get the area of a triangle]

    Args:
        a ([numpy float array]): [the place of point a]
        b ([numpy float array]): [the place of point b]
        c ([numpy float array]): [the place of point c]

    Returns:
        area [float]: [area]
    """
    dist_a = get_dist(a, b)
    dist_b = get_dist(b, c)
    dist_c = get_dist(c, a)

    area = sqrt((dist_a + dist_b + dist_c) * (dist_a + dist_b - dist_c) * (dist_a - dist_b + dist_c) * (-dist_a + dist_b + dist_c)) / 4
    return area

def get_average_dist_and_area(full_file_name):
    """[get average area of original mesh]

    Args:
        full_file_name ([str]): [the full filename of original mesh]

    Returns:
        avg_dist [dist]: [the average dist of the lines in the original mesh]
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
    my_vertexs = np.array(my_vertexs, dtype = 'float32')
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
    """[get the pairs of the points that have lines in a mesh]

    Args:
        faces ([numpy int array]): [the faces]
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
    """[split a line into several points]

    Args:
        a ([numpy float array]): [the place of point a]
        b ([numpy float array]): [the place of point b]
        avg_dist ([float]): [the average dist of lines]

    Returns:
        new_points: [numpy float array]: [the place of new points]
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
    """[split a line into several points]

    Args:
        a ([numpy float array]): [the place of point a]
        b ([numpy float array]): [the place of point b]
        c ([numpy float array]): [the place of point c]
        avg_area ([float]): [the average dist of areas]

    Returns:
        new_points: [numpy float array]: [the place of new points]
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
    """[get the background pointcloud based on the original info]

    Args:
        base_dir_scannet ([str]): [the base directory of ScanNet dataset]
        base_dir_plane ([str]): [the base directory of ScanNet-Planes dataset]
        scene_id ([str]): [the scene id]
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
    
    new_points = np.array(new_points, dtype = 'float32')
    result_vertexs = np.concatenate((vertexs, new_points), axis = 0)
    
    write_pointcloud_file(save_name_ply, result_vertexs)
    print('written', save_name_ply)

def main():
    """[the main function]
    """
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--base_dir_scannet', default = '/home1/shenguanlin/scannet_pretrain', type = str)
    parser.add_argument('--base_dir_plane', default = '/home1/shenguanlin/scannet_planes', type = str)
    args = parser.parse_args()
    full_name_list = glob.glob(os.path.join(args.base_dir_plane, '*.ply'))
    for full_name in full_name_list:
        scene_id = full_name.split(os.sep)[-1][:-4]
        print('Rendering', scene_id)
        get_background_pointcloud(args.base_dir_scannet, args.base_dir_plane, scene_id)

if __name__ == "__main__":
    main()
