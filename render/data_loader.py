import numpy as np
import skimage.io as sio


def LoadPLY(model_path):
    '''
    description: load the obj file with only points with color, vertexs, no normal or texture
    input: model_path
    return: vertexs, vertex_colors, faces
    '''
    f = open(model_path, 'r')
    lines = f.read().split('\n')
    f.close()
    vertex_num = int(lines[2].split()[-1])
    face_num = int(lines[9].split()[-1])
    my_vertexs = []
    my_vertex_colors = []
    my_faces = []
    for i in range(vertex_num):
        line = lines[12 + i]
        words = line.split()
        x = float(words[0])
        y = float(words[1])
        z = float(words[2])
        r = float(words[3])
        g = float(words[4])
        b = float(words[5])
        my_vertexs.append([x, y, z])
        my_vertex_colors.append([r / 255, g / 255, b / 255])
    for i in range(face_num):
        line = lines[12 + vertex_num + i]
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
