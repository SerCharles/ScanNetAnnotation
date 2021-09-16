"""The main function of vanishing point annotation
"""
import os
import numpy as np
import utils
import data_utils
import time

def annotation_one_picture():
    intrinsic, extrinsic, layout_seg = data_utils.load_data('/home1/sgl/scannet_mine', 'scene0000_01', 0)
    H, W = layout_seg.shape
    start = time.time()
    vy, vx = utils.get_vanishing_point(H, W, intrinsic, extrinsic)
    print(vy, vx)
    lines = utils.get_lines(H, W, vy, vx)
    whether_ceilings, whether_floors, whether_walls, ceiling_places, floor_places = utils.get_ceiling_and_floor(layout_seg, lines, 7, 4)
    data_utils.visualize_annotation_result(layout_seg, whether_ceilings, whether_floors, whether_walls, ceiling_places, floor_places)
    end = time.time()
    print(end - start)
annotation_one_picture()