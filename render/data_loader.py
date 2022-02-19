"""The data loading and processing function used in cuda rendering 
"""

import numpy as np
import os
from plyfile import *
from math import *
import json

def transform_norm(norm, world2cam):
    """Transform the normals in the world coordinate to a camera's coordinate, normalize it and ensure that it points to the eye
        V: number of vertexs
        F: number of faces

    Args:
        norm [numpy float array], [V * 3]: [the normal in the world coordinate]
        world2cam [numpy float array], [3 * 3]: [the transform matrix between the world coordinate and the camera coordinate]

    Returns:
        real_norm [numpy float array], [V * 3]: [the modified norm in the camera coordinate, added by 1 to be saved as pictures]
    """
    rotate = world2cam[0:3, 0:3]
    norm = np.swapaxes(norm, 0, 1)
    norm_cam = np.dot(rotate, norm)
    norm_cam = np.swapaxes(norm_cam, 0, 1)
    whether_z_positive = np.repeat((norm_cam[:, 2:3] > 0), 3, axis = 1)
    whether_z_negative = ~whether_z_positive
    real_norm = norm_cam * whether_z_negative + (-norm_cam) * whether_z_positive
    real_norm = real_norm + 1.0
    return real_norm

def load_intrinsic(full_name):
    """Load one intrinsic file

    Args:
        full_name [string]: [the full path of the intrinsic file]

    Returns:
        width [int]: [the width of the picture]
        height [int]: [the height of the picture]
        fx [float]: [the focal length of the x axis, part of the intrinsic]
        fy [float]: [the focal length of the y axis, part of the intrinsic]
        cx [float]: [the focal center of the x axis, part of the intrinsic]
        cy [float]: [the focal center of the y axis, part of the intrinsic]
    """
    width = 0
    height = 0
    fx = 0.0
    fy = 0.0 
    cx = 0.0
    cy = 0.0
    f = open(full_name, 'r')
    lines = f.read().split('\n')
    for line in lines: 
        words = line.split()
        if len(words) > 0:
            if words[0] == 'm_colorWidth':
                width = int(words[2])
            elif words[0] == 'm_colorHeight':
                height = int(words[2])
            elif words[0] == 'm_calibrationColorIntrinsic':
                fx = float(words[2])
                fy = float(words[7])
                cx = float(words[4])
                cy = float(words[8])
    f.close()
    return width, height, fx, fy, cx, cy

def load_pose(full_name):
    """Load one extrinsic file(camera to world)

    Args:
        full_name [string]: [the full path of the extrinsic file]

    Returns:
        pose [numpy float array], [4 * 4]: [the extrinsic matrix]
    """
    pose = np.zeros((4, 4), dtype = np.float32)
    f = open(full_name, 'r')
    lines = f.read().split('\n')
    
    for i in range(4):
        words = lines[i].split()
        for j in range(4):
            word = float(words[j])
            pose[i][j] = word

    pose = pose.astype(np.float32)
    f.close()
    return pose


def load_simple_ply(model_path):
    """Load a simple ply file with on point places and faces
        V: number of vertexs
        F: number of faces

    Args:
        model_path [string]: [the full path of the model]

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
    
    vertexs = np.array(my_vertexs, dtype='float32')
    faces = np.array(my_faces, dtype='int32')
    return vertexs, faces


def load_ply(model_path):
    """Load the ply file with points with normal and color, and faces
        V: number of vertexs
        F: number of faces

    Args:
        model_path [string]: [the full path of the model]

    Returns:
        vertexs [numpy float array], [V * 3]: [vertexs]
        faces [numpy int array], [F * 3]: [faces]
        norms [numpy float array], [V * 3]: [norms of each vertex]
        colors [numpy float array], [V * 3]: [colors of each vertex, divided by 256]
    """
    plydata = PlyData.read(model_path)
    my_vertexs = []
    my_norms = []
    my_colors = []
    my_faces = []

    vertexs = plydata['vertex']
    faces = plydata['face']
    for i in range(vertexs.count):
        x = float(vertexs[i][0])
        y = float(vertexs[i][1])
        z = float(vertexs[i][2])
        nx = float(vertexs[i][3])
        ny = float(vertexs[i][4])
        nz = float(vertexs[i][5])
        r = float(vertexs[i][6]) / 256
        g = float(vertexs[i][7]) / 256
        b = float(vertexs[i][8]) / 256

        my_vertexs.append([x, y, z])
        my_colors.append([r, g, b])
        my_norms.append([nx, ny, nz])


    for i in range(faces.count):
        a = int(faces[i][0][0])
        b = int(faces[i][0][1])
        c = int(faces[i][0][2])
        my_faces.append([a, b, c])
    
    vertexs = np.array(my_vertexs, dtype='float32')
    faces = np.array(my_faces, dtype='int32')
    norms = np.nan_to_num(np.array(my_norms, dtype='float32'))
    colors = np.array(my_colors, dtype='float32')
    return vertexs, faces, norms, colors


def load_ply_without_norm(model_path):
    """Load the ply file with points with color, and faces
        V: number of vertexs
        F: number of faces

    Args:
        model_path [string]: [the full path of the model]

    Returns:
        vertexs [numpy float array], [V * 3]: [vertexs]
        faces [numpy int array], [F * 3]: [faces]
        colors [numpy float array], [V * 3]: [colors of each vertex, divided by 256]
    """
    plydata = PlyData.read(model_path)
    my_vertexs = []
    my_colors = []
    my_faces = []

    vertexs = plydata['vertex']
    faces = plydata['face']
    for i in range(vertexs.count):
        x = float(vertexs[i][0])
        y = float(vertexs[i][1])
        z = float(vertexs[i][2])
        r = float(vertexs[i][3]) / 256
        g = float(vertexs[i][4]) / 256
        b = float(vertexs[i][5]) / 256

        my_vertexs.append([x, y, z])
        my_colors.append([r, g, b])


    for i in range(faces.count):
        a = int(faces[i][0][0])
        b = int(faces[i][0][1])
        c = int(faces[i][0][2])
        my_faces.append([a, b, c])
    
    vertexs = np.array(my_vertexs, dtype='float32')
    faces = np.array(my_faces, dtype='int32')
    colors = np.array(my_colors, dtype='float32')
    return vertexs, faces, colors



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
