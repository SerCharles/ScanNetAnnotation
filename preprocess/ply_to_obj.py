import os
import json
import numpy as np
from plyfile import *

def ply_to_obj(ROOT_FOLDER, scene_id):
    '''
    description: switch the ply file to obj, in order to render
    parameters: root folder of scannet, the scene id
    return: empty, write the obj file
    '''
    ply_name = os.path.join(ROOT_FOLDER, scene_id, 'annotation', 'planes.ply')
    obj_name = os.path.join(ROOT_FOLDER, scene_id, 'annotation', 'planes.obj')
    plydata = PlyData.read(ply_name)
    vertexs = plydata['vertex']
    faces = plydata['face']
    try:
        edges = plydata['edge']
    except: 
        edges = []
    file_out = open(obj_name, 'w')
    for vertex in vertexs:
        x, y, z, r, g, b = vertex
        file_out.write('v ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(r / 255) + ' ' + str(g / 255) + ' ' + str(b / 255) + '\n')
    for face in faces: 
        a, b, c = face[0]
        file_out.write('f ' + str(a + 1) + ' ' + str(b + 1) + ' ' + str(c + 1) + '\n')
    for edge in edges: 
        a, b, r, g, bb = edge 
        file_out.write('l ' + str(a + 1) + ' ' + str(b + 1) + '\n')
    file_out.close()



ply_to_obj("E:\\dataset\\scannet\\scans", 'scene0000_00')