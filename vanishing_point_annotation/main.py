"""The main function of vanishing point annotation
"""
import os
import numpy as np
import utils
import data_utils

def annotation_one_picture():
    intrinsic, extrinsic, layout_seg = data_utils.load_data('/home1/sgl/scannet_mine', 'scene0000_01', 0)
    H, W = layout_seg.shape
    vx, vy = utils.get_vanishing_point(H, W, intrinsic, extrinsic)
    print(vx, vy)
    
annotation_one_picture()