import numpy as np
import os
from math import *
from get_parameter_geolayout import *

def parameter_loss(parameter, parameter_gt):
    '''
    description: get the parameter loss 
    parameter: the parameter driven by our model, the ground truth parameter
    return: parameter loss
    '''
    p, q, r, s = parameter
    p_gt, q_gt, r_gt, s_gt = parameter_gt
    dp = (np.abs(p - p_gt)).sum()
    dq = (np.abs(q - q_gt)).sum() 
    dr = (np.abs(r - r_gt)).sum() 
    ds = (np.abs(s - s_gt)).sum() 
    loss = dp + dq + dr + ds 
    return loss 

def discrimitive_loss(parameter, plane_seg_gt, plane_id_gt, average_plane_info, delta_v, delta_d):
    '''
    description: get the discrimitive loss 
    parameter: the parameter driven by our model, the ground truth segmentation loss, the ground truth plane id
        the average plane info, threshold of the same plane, threshold of different planes
    return: depth loss
    '''
    p, q, r, s = parameters
    C = len(plane_id_gt)
    lvars = {}
    lvar = 0
    dvar = 0
    for i in range(C):
        t_id = plane_id_gt[i]
        lvars[t_id] = {'count': 0, 'loss': 0.0} 
    
    #get lvars
    for v in range(len(plane_seg)) :
        for u in range(len(plane_seg[v])): 
            the_seg = plane_seg[v][u] 
            the_p = p[v][u]
            the_q = q[v][u]
            the_r = r[v][u]
            the_s = s[v][u]
            gt_p = average_plane_info[the_seg]['p']
            gt_q = average_plane_info[the_seg]['q']
            gt_r = average_plane_info[the_seg]['r']
            gt_s = average_plane_info[the_seg]['s']

            dp = abs(the_p - gt_p)
            dq = abs(the_q - gt_q)
            dr = abs(the_r - gt_r)
            ds = abs(the_s - gt_s)

            loss = max(0, dp + dq + dr + ds - delta_v)
            lvars[the_seg]['count'] += 1 
            lvars[the_seg]['loss'] += loss 
    for the_id in plane_id_gt: 
        lvar += (lvars[the_id]['loss'] / lvars[the_id]['count'])
    lvar /= C 

    #get dvar 
    for i in range(C - 1):
        for j in range(i + 1, C):
            id_i = plane_id_gt[i]
            id_j = plane_id_gt[j]
            pi = average_plane_info[id_i]['p']
            qi = average_plane_info[id_i]['q']
            ri = average_plane_info[id_i]['r']
            si = average_plane_info[id_i]['s']
            pj = average_plane_info[id_j]['p']
            qj = average_plane_info[id_j]['q']
            rj = average_plane_info[id_j]['r']
            sj = average_plane_info[id_j]['s']

            dp = abs(pi - pj)
            dq = abs(qi - qj) 
            dr = abs(ri - rj)
            ds = abs(si - sj)

            loss = max(0, delta_d - dp - dq - dr - ds)
            dvar += loss 
    dvar /= (C * (C - 1))
    return lvar + dvar
  

def depth_loss(parameter, plane_id_gt, average_plane_info, depth_gt):
    '''
    description: get the depth loss 
    parameter: the depth driven by our model, the ground truth depth
    return: depth loss
    '''
    average_plane_info = get_average_depth_map(plane_id_gt, average_plane_info, depth_gt.shape)
    loss = 0
    for the_id in plane_id_gt:
        depth = average_plane_info[the_id]['depth_map']
        new_loss = (np.abs(depth - depth_gt)).sum()
        loss += new_loss
    loss /= len(plane_id_gt)
    return loss