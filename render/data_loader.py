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
    print(norm.shape)
    norm = np.swapaxes(norm, 0, 1)
    norm_cam = np.dot(rotate, norm)
    norm_cam = np.swapaxes(norm_cam, 0, 1)
    whether_z_positive = np.repeat((norm_cam[:, 2:3] > 0), 3, axis = 1)
    whether_z_negative = ~whether_z_positive
    real_norm = norm_cam * whether_z_negative + (-norm_cam) * whether_z_positive
    real_norm = real_norm + 1.0
    return real_norm





def LoadPLY(model_path):
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


def LoadOBJ(model_path):
    '''
    description: load the obj file with only points with color, vertexs, no normal or texture
    input: model_path
    return: vertexs, vertex_colors, faces
    '''
    vertexs = []
    vertex_colors = []
    faces = []
    lines = [l.strip() for l in open(model_path)]

    for l in lines:
        words = [w for w in l.split(' ') if w != '']
        if len(words) == 0:
            continue

    if words[0] == 'v':
        vertexs.append([float(words[1]), float(words[2]), float(words[3])])
        vertex_colors.append([float(words[4]), float(words[5]), float(words[6])])

    elif words[0] == 'f':
        f = []
        for j in range(1, len(words)):
            f.append(int(words[j]))
        faces.append(f)

    F = np.array(faces, dtype = 'int32')
    V = np.array(vertexs, dtype = 'float32')
    V = (V * 0.5).astype('float32')
    VC = np.array(vertex_colors, dtype = 'float32')

    return V, VC, F
