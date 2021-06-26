'''
Get the full mesh of the background
'''
import numpy as np
import os
from plyfile import *
from math import *
import json


def get_area(a, b, c):
    """[get the distance of a triangle]

    Args:
        a ([numpy float array]): [the place of point a]
        b ([numpy float array]): [the place of point b]
        c ([numpy float array]): [the place of point c]

    Returns:
        float [float]: [area]
    """
    dist_a = sqrt(float(np.dot(b - a, b - a)))
    dist_b = sqrt(float(np.dot(c - b, c - b)))
    dist_c = sqrt(float(np.dot(a - c, a - c)))

    p = (dist_a + dist_b + dist_c) / 2

    dist = sqrt(p(p - a)(p - b)(p - c))
    return dist

def get_average_distance(full_file_name):
    """[get average area of original mesh]

    Args:
        full_file_name ([str]): [the full filename of original mesh]

    Returns:
        avg_area [float]: [the average area of the triangles in the original mesh]
    """

    plydata = PlyData.read(full_file_name)
    my_vertexs = []
    total_area = 0.0 
    total_num = 0

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
        total_area += get_area(my_vertexs[a], my_vertexs[b], my_vertexs[c])
        total_num += 1

    avg_area = total_area / total_num
    return avg_area


#TODO:根据当前mesh面积/最小单元面积划定等分数量，用等分法生成mesh
def split_line_points():
    pass

def split_inner_points():
    pass

def split_one_mesh(vertexs, faces):
    """[summary]

    Args:
        vertexs ([type]): [description]
        faces ([type]): [description]
    """

def get_background_mesh(base_dir, scene_id):
    """[get the background mesh based on the original info]

    Args:
        base_dir ([str]): [the base directory of ScanNet-Planes dataset]
        scene_id ([str]): [the scene id]
    """
    full_name_ply = os.path.join(base_dir, scene_id + '.ply')
    save_name_ply = os.path.join(base_dir, scene_id + '_full.ply')




