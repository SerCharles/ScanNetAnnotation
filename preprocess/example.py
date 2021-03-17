import sys
import numpy as np
import sys
import skimage.io as sio
import os
import shutil
from myloader import LoadOBJ

libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(libpath[:-4], 'lib'))
import render
from myloader import *


input_obj = '/home/shenguanlin/scannet/scans/scene0000_00/annotation/planes.obj'
input_pose = '/home/shenguanlin/scannet/scans/scene0000_00/pose'
interval = 30
V, VC, F, L = LoadOBJ(input_obj)

poses = LoadPoses(input_pose, interval)

for ii in range(len(poses)):
    pose = poses[ii]
    pose_id = interval * ii

    # set up camera information
    info = {'Height':480, 'Width':640, 'fx':575, 'fy':575, 'cx':319.5, 'cy':239.5}
    render.setup(info)

    # set up mesh buffers in cuda
    context = render.SetMesh(V, F)



    world2cam = np.linalg.inv(pose).astype('float32')

    # the actual rendering process
    render.render(context, world2cam)

    # get depth information
    depth = render.getDepth(info)

    # get information of mesh rendering
    # vindices represents 3 vertices related to pixels
    # vweights represents barycentric weights of the 3 vertices
    # findices represents the triangle index related to pixels
    vindices, vweights, findices = render.getVMap(context, info)

    x_shape = findices.shape[0]
    y_shape = findices.shape[1]
    final_color = np.zeros((x_shape, y_shape, 3), dtype='float32')
    for i in range(x_shape):
        for j in range(y_shape):
            current_point_face = findices[i][j]
            if current_point_face == 0:
                continue 
            current_point_indices = F[current_point_face]
            current_point_weights = vweights[i][j]

            for k in range(len(current_point_indices)):
                indice = current_point_indices[k] - 1
                weight = current_point_weights[k]
                color = VC[indice]
                r = color[0]
                g = color[1]
                b = color[2]
                final_color[i][j][0] += r * weight
                final_color[i][j][1] += g * weight 
                final_color[i][j][2] += b * weight

    render.Clear()
    if not os.path.exists('result'):
        os.mkdir('result')
    sio.imsave(os.path.join('result', str(pose_id) + '.png'), final_color)
