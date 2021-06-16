'''
Generate the ply mesh file and relative info based on the json data
'''
import numpy as np
import os
from math import *
import json

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

def generate_mesh(base_dir, scene_id):
    """[generate the mesh of the data based on the info data]

    Args:
        base_dir ([str]): [the base directory of ScanNet-Planes dataset]
        scene_id ([str]): [the scene id]
    """
    full_name_ply = os.path.join(base_dir, scene_id + '.ply')
    full_name_ply = 'C:\\Users\\Lenovo\\Desktop\\kebab.ply'
    full_name_info = os.path.join(base_dir, scene_id + '.json')
    with open(full_name_info, 'r', encoding = 'utf8')as fp:
        data = json.load(fp)
    vertexs = data['verts']
    planes = data['quads']
    faces = []
    face_labels = []
    face_norms = []

    for i in range(len(planes)):
        plane = planes[i]
        for j in range(len(plane) - 2):
            new_face = [plane[0], plane[j + 1], plane[j + 2]]
            faces.append(new_face)
            face_labels.append(i)

        #TODO:最小二乘求法向
    write_ply_file(full_name_ply, vertexs, faces)

