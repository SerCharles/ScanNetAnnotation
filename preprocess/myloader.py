import numpy as np
import skimage.io as sio
import glob
import os
import functools

def compare_filename(name_a, name_b):
    id_a = int(name_a.split(os.sep)[-1][:-4])
    id_b = int(name_b.split(os.sep)[-1][:-4])
    return id_a - id_b 


def LoadPoses(pose_path, interval):
    '''
    description: load the pose files to be rendered
    input: pose_path, render interval
    return: the list of poses
    '''
    pose_files = glob.glob(os.path.join(pose_path, '*.txt'))
    pose_files.sort(key = functools.cmp_to_key(compare_filename))
    poses = []
    for i in range(len(pose_files)):
        if i % interval == 0:
            pose = []
            file_name = pose_files[i]
            lines = [l.strip() for l in open(pose_files[i])]
            for j in range(len(lines)):
                l = lines[j]
                words = l.split()
                if len(words) == 0:
                    continue
                pose.append([])
                for word in words:
                    word = float(word)
                    pose[j].append(word)
            pose = np.array(pose, dtype='float32')
            poses.append(pose)
    return poses


def LoadOBJ(model_path):
    '''
    description: load the obj file with only points with color, vertexs and lines, no normal or texture
    input: model_path
    return: vertexs, vertex_colors, faces, lines
    '''
    vertexs = []
    vertex_colors = []
    faces = []
    edges = []
    lines = [l.strip() for l in open(model_path)]

    for l in lines:
        words = l.split()
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
        elif words[0] == 'l':
            l = []
            for j in range(1, len(words)):
                l.append(int(words[j]))
            edges.append(l)

    F = np.array(faces, dtype = 'int32')
    L = np.array(edges, dtype = 'int32')
    V = np.array(vertexs, dtype = 'float32')
    V = (V * 0.5).astype('float32')
    VC = np.array(vertex_colors, dtype = 'float32')

    return V, VC, F, L
