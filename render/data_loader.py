import numpy as np
import skimage.io as sio
from plyfile import *
from math import *

from skimage.util import dtype

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

