import sys
import numpy as np
import sys
import skimage.io as sio
import os
import shutil
from objloader import LoadOBJ

libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(libpath[:-4], 'lib'))
import render
import objloader


input_obj = '/home/shenguanlin/scannet/scans/scene0000_00/annotation/planes_with_line.obj'
V, VC, F, L = LoadOBJ(input_obj)

# set up camera information
info = {'Height':480, 'Width':640, 'fx':575, 'fy':575, 'cx':319.5, 'cy':239.5}
render.setup(info)

# set up mesh buffers in cuda
context = render.SetMesh(V, F)

cam2world = np.array([[-0.955421, 0.119616, -0.269932, 2.655830],
[0.295248, 0.388339, -0.872939, 2.981598],
[0.000408, -0.913720, -0.406343, 1.368648],
[0.000000, 0.000000, 0.000000, 1.000000]], dtype = np.float32)

world2cam = np.linalg.inv(cam2world).astype('float32')

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
        current_point_indices = vindices[i][j]
        current_point_weights = vweights[i][j]

        for k in range(len(current_point_indices)):
            indice = current_point_indices[k]
            weight = current_point_weights[k]
            color = VC[indice - 1]
            r = color[0]
            g = color[1]
            b = color[2]
            final_color[i][j][0] += r * weight
            final_color[i][j][1] += g * weight 
            final_color[i][j][2] += b * weight

render.Clear()
sio.imsave('result.png', final_color)
