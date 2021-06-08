import numpy as np
import skimage.io as sio
from plyfile import *

def LoadPLY(model_path):
    '''
    description: load the ply file with only points with color, vertexs, and normal
    input: model_path
    return: vertexs, vertex_colors, faces
    '''
    f = open(model_path, 'r')
    lines = f.read().split('\n')
    f.close()
        
    vertex_num = int(lines[3].split()[-1])
    face_num = int(lines[14].split()[-1])
    my_vertexs = []
    my_vertex_norms = []
    my_vertex_colors = []
    my_faces = []
    for i in range(vertex_num):
        line = lines[17 + i]
        words = line.split()
        x = float(words[0])
        y = float(words[1])
        z = float(words[2])
        nx = float(words[3])
        ny = float(words[4])
        nz = float(words[5])
        r = float(words[6])
        g = float(words[7])
        b = float(words[8])
        my_vertexs.append([x, y, z])
        my_vertex_norms.append([nx, ny, nz])
        my_vertex_colors.append([r / 255, g / 255, b / 255])
    for i in range(face_num):
        line = lines[17 + vertex_num + i]
        words = line.split()
        a = int(words[1])
        b = int(words[2])
        c = int(words[3])
        f = [a, b, c]
        my_faces.append(f)
    
    F = np.array(my_faces, dtype = 'int32')
    V = np.array(my_vertexs, dtype = 'float32')
    VC = np.array(my_vertex_colors, dtype = 'float32')
    return V, VC, F

def LoadDensePLY(model_path):
    '''
    description: load the zipped ply file with only points with color, vertexs, and normal
    input: model_path
    return: vertexs, vertex_colors, faces
    '''
    plydata = PlyData.read(model_path)
    my_vertexs = []
    my_vertex_norms = []
    my_vertex_colors = []
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
        r = float(vertexs[i][6])
        g = float(vertexs[i][7])
        b = float(vertexs[i][8])
        my_vertexs.append([x, y, z])
        my_vertex_norms.append([nx, ny, nz])
        my_vertex_colors.append([r / 255, g / 255, b / 255])

    for i in range(faces.count):
        a = int(faces[i][0][0])
        b = int(faces[i][0][1])
        c = int(faces[i][0][2])
        my_faces.append([a, b, c])
    F = np.array(my_faces, dtype = 'int32')
    V = np.array(my_vertexs, dtype = 'float32')
    VC = np.array(my_vertex_colors, dtype = 'float32')
    return V, VC, F


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
