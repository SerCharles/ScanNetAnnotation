import numpy as np
import os
from plyfile import *
from math import *
import json

def transform_norm(norm, world2cam):
    '''
    description: transform, normalize and set positive the norm
    parameter: norm, world2cam
    return: new norm
    '''
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
    '''
    description: read the intrinsic
    parameter: full_name
    return: width, height, fx, fy, cx, cy
    '''
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
    '''
    description: read the extrinsic
    parameter: full_name
    return: numpy array of extrinsic
    '''
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
    '''
    description: load the simple ply file with only basic vertex, face info
    input: model_path
    return: vertexs, faces, norm, color
    '''
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


def load_ply(model_path):
    '''
    description: load the zipped ply file with only points with color, vertexs, and normal
    input: model_path
    return: vertexs, faces, norm, color
    '''
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
    
    F = np.array(my_faces, dtype = 'int32')
    V = np.array(my_vertexs, dtype = 'float32')
    C = np.array(my_colors, dtype = 'float32')
    NORM = np.nan_to_num(np.array(my_norms, dtype = 'float32'))

    return V, F, NORM, C



def load_planes(base_dir, scene_id):
    """[load the ScanNet-Planes dataset]

    Args:
        base_dir ([str]): [the base directory of ScanNet-Planes dataset]
        scene_id ([str]): [the scene id]
    

    Returns:
        V [float32 numpy array]: [vertexs]
        F [int32 numpy array]: [faces]
        NORM [float32 numpy array]: [norms of each face]
        L [int32 numpy array]: [labels of each face]

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
    with open(full_name_info, 'r', encoding = 'utf8')as fp:
        data = json.load(fp)
    norms = data['norms']
    labels = data['labels']
    F = np.array(faces, dtype = 'int32')
    V = np.array(vertexs, dtype = 'float32')
    L = np.array(labels, dtype = 'int32')
    NORM = np.nan_to_num(np.array(norms, dtype = 'float32'))

    return V, F, NORM, L




def get_seg_label(base_dir, scene_id, vertexs, faces):
    """[get the segmentation label of the model]

    Args:
        base_dir ([str]): [the base dir of the ScanNet dataset]
        scene_id ([str]): [the scene id ]
        vertexs ([ply vertexs]): [vertexs info generated by plyfile]
        faces ([ply faces]): [faces info generated by plyfile]

    Returns:
        face labels [boolean array]: [whether the face is background, True is yes, False is no]
    """

    background_labels = ['ceiling', 'floor', 'wall']
    instance_descriptor_name = os.path.join(base_dir, scene_id, 'ply', scene_id + '_vh_clean.aggregation.json') 
    vertex_info_name = os.path.join(base_dir, scene_id, 'ply', scene_id + '_vh_clean_2.0.010000.segs.json')

    with open(instance_descriptor_name, 'r', encoding = 'utf8')as fp:
        instance_info = json.load(fp)
    instances = instance_info['segGroups']

    with open(vertex_info_name, 'r', encoding = 'utf8')as fp:
        vertex_info_total = json.load(fp)
        vertex_info = vertex_info_total['segIndices']

    vertex_label = []
    face_label = []

    for i in range(len(vertexs)):
        vertex_label.append(False)
    for i in range(len(faces)):
        face_label.append(False)

    for i in range(len(vertexs)):
        representative = int(vertex_info[i])
        for instance in instances:
            segments = instance['segments']
            label = instance['label']
            if representative in segments:
                if label in background_labels:
                    vertex_label[i] = True
                break 
    
    for i in range(len(faces)):
        a = int(faces[i][0])
        b = int(faces[i][1])
        c = int(faces[i][2])
        if vertex_label[a] == True and vertex_label[b] == True and vertex_label[c] == True:
            face_label[i] = True

    return face_label