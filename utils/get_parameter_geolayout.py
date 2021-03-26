import numpy as np
import os
from math import *
import skimage.io as io

def get_parameter(depth_map):
    '''
    description: get the ground truth parameters(p, q, r, s) from the depth map
    parameter: depth map
    return: the (p, q, r, s) value of all pixels
    '''
    p = np.zeros(depth_map.shape, dtype = float)  
    q = np.zeros(depth_map.shape, dtype = float)  
    r = np.zeros(depth_map.shape, dtype = float)  
    s = np.zeros(depth_map.shape, dtype = float) 
    for v in range(len(depth_map)):
        for u in range(len(depth_map[v])): 
            zi = float(depth_map[v][u]) 

            if u != len(depth_map[v]) - 1:
                pi = float(depth_map[v][u + 1]) - float(depth_map[v][u])
            else: 
                pi = p[v][u - 1]
            if v != len(depth_map) - 1:
                qi = float(depth_map[v + 1][u]) - float(depth_map[v][u])
            else: 
                qi = q[v - 1][u]
            ri = 1 / zi - pi * u - qi * v
            si = sqrt(pi ** 2 + qi ** 2 + ri ** 2) 
            pi /= si 
            qi /= si 
            ri /= si 
            p[v][u] = pi 
            q[v][u] = qi 
            r[v][u] = ri 
            s[v][u] = si 
    return p, q, r, s 

def get_depth_map(p, q, r, s):
    '''
    description: get the depth map from the parameters(p, q, r, s)
    parameter: the (p, q, r, s) value of all pixels
    return: evaluated depth map
    '''
    depth_map = np.zeros(p.shape) 
    for v in range(len(depth_map)):
        for u in range(len(depth_map[v])): 
            depth_map[v][u] = 1 / ((p[v][u] * u + q[v][u] * v + r[v][u]) * s[v][u])
    return depth_map

def get_average_plane_info(parameters, plane_seg, plane_ids):
    '''
    description: get the average plane info 
    parameter: parameters per pixel, plane segmentation per pixel, the list of plane ids 
    return: average plane info
    '''
    average_paramater = {}
    for t_id in plane_ids: 
        average_paramater[t_id] = {'count': 0, 'p': 0.0, 'q': 0.0, 'r': 0.0, 's': 0.0}
    p, q, r, s = parameters
    for v in range(len(plane_seg)) :
        for u in range(len(plane_seg[v])): 
            the_seg = plane_seg[v][u] 
            the_p = p[v][u]
            the_q = q[v][u]
            the_r = r[v][u]
            the_s = s[v][u]
            average_paramater[the_seg]['count'] += 1
            average_paramater[the_seg]['p'] += the_p
            average_paramater[the_seg]['q'] += the_q
            average_paramater[the_seg]['r'] += the_r
            average_paramater[the_seg]['s'] += the_s
    for the_id in plane_ids: 
        average_paramater[the_id]['p'] /= average_paramater[the_id]['count']
        average_paramater[the_id]['q'] /= average_paramater[the_id]['count']
        average_paramater[the_id]['r'] /= average_paramater[the_id]['count']
        average_paramater[the_id]['s'] /= average_paramater[the_id]['count']
    return average_paramater

def get_average_depth_map(plane_ids, average_plane_info, shape):
    '''
    description: get the depth from the average parameters(p, q, r, s)
    parameter: the plane ids, the average plane infos, the shape of the depth map
    return: evaluated depth maps of all planes
    '''
    for the_id in plane_id_gt: 
        p = average_plane_info[the_id]['p']
        q = average_plane_info[the_id]['q']
        r = average_plane_info[the_id]['r']
        s = average_plane_info[the_id]['s']
        depth_map = np.zeros(shape) 
        for v in range(len(depth_map)):
            for u in range(len(depth_map[v])): 
                depth_map[v][u] = 1 / ((p * u + q * v + r) * s)
        average_plane_info[the_id]['depth_map'] = depth_map
    return average_plane_info

'''
name = 'E:\\dataset\\geolayout\\training\\layout_depth\\00af93c06521455ea528309996881b8d_i1_5_layout.png'
depth_map_original = io.imread(name)
p, q, r, s = get_parameter_geolayout(depth_map_original)
depth_map = get_depth_map_geolayout(p, q, r, s)
print(p.shape, q.shape, r.shape, s.shape) 
print(depth_map_original.shape, depth_map.shape)
rate = 0
for v in range(len(depth_map)):
    for u in range(len(depth_map[v])): 
        if float(depth_map[v][u]) - float(depth_map_original[v][u]) < 0.5:  
            rate += 1 
rate /= (len(depth_map) * len(depth_map[0]))
print(rate)
'''