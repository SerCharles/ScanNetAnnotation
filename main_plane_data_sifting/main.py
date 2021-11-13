"""The main functions of main plane sifting
"""
import os
import glob
import argparse
import time
import numpy as np
from math import *
from numpy.core.fromnumeric import mean
import torch

import data_utils, utils

def process_one_picture(base_dir, scene_id, id, save_dir):
    """Process one picture
        H: the height of the picture
        W: the width of the picture
        
    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the scene id to be handled]
        id [int]: [the id of the picture]
        save_dir [string]: [the save directory]
    """
    start = time.time()
    base_name, intrinsic, extrinsic, image, depth, normal, seg, layout_seg, vanishing_point, whether_boundary, mask\
        = data_utils.load_one_picture(base_dir, scene_id, id)
    _, H, W = image.size()
    plane_info_per_pixel = utils.get_plane_info_per_pixel(normal, depth, intrinsic)
    cluster_result = utils.clustering(seg, plane_info_per_pixel)
    cluster_ids = torch.unique(cluster_result)
    average_plane_info = utils.get_average_plane_info_from_pixels(plane_info_per_pixel, cluster_result)
    average_plane_info_global = utils.rotate_plane(average_plane_info, source_extrinsic=extrinsic)

    #classify planes  
    ceiling_ids = []
    floor_ids = []
    wall_ids = []
    for i in range(len(cluster_ids)):
        plane_id = int(cluster_ids[i])
        if plane_id == 0:
            continue
        plane_info = average_plane_info_global[plane_id]
        A = plane_info[0]
        B = plane_info[1]
        C = plane_info[2]
        D = plane_info[3]
        if C < -0.5:
            ceiling_ids.append(plane_id)
        elif C > 0.5:
            floor_ids.append(plane_id)
        else: 
            wall_ids.append(plane_id)
    
    whether_valid = False
    wall_segs = []
    bounding_boxes = []
    estimate_boundaries = []
    if len(wall_ids) > 0:        
        whether_valid = True        
        for plane_id in wall_ids:
            wall_seg = torch.eq(cluster_result, plane_id) #1 * H * W
            bounding_box = utils.get_bounding_box(vanishing_point, wall_seg)
            estimate_boundary = utils.estimate_boundary(bounding_box, whether_boundary)
            
            wall_seg = wall_seg.unsqueeze(0)
            bounding_box = bounding_box.unsqueeze(0)
            estimate_boundary = estimate_boundary.unsqueeze(0)
            wall_segs.append(wall_seg) 
            bounding_boxes.append(bounding_box)
            estimate_boundaries.append(estimate_boundary)
            
        wall_segs = torch.cat(wall_segs, dim=0) #M * 1 * H * W
        bounding_boxes = torch.cat(bounding_boxes, dim=0) #M * 2
        estimate_boundaries = torch.cat(estimate_boundaries, dim=0) #M * 2
        mean_dist = utils.boundary_dist(bounding_boxes, estimate_boundaries, W)

        data_utils.save_one_picture(base_dir, scene_id, id, save_dir, image, vanishing_point, whether_boundary, wall_segs, bounding_boxes, estimate_boundaries)
    
    else: 
        mean_dist = 0.0
    
    end = time.time()
    print('Processed', base_name, 'time cost:', '{:.4f}s'.format(end - start), 'mean dist:', '{:.4f}'.format(mean_dist))
    return whether_valid, mean_dist


def process_one_scene(base_dir, scene_id, save_dir):
    """Process one scene

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the scene id to be handled]
        save_dir [string]: [the save directory]
    """
    print('Processing', scene_id)
    valid_id_list = []
    
    full_name_list = glob.glob(os.path.join(base_dir, scene_id, 'pose', "*.txt"))
    id_list = []
    for full_name in full_name_list:
        file_name = full_name.split(os.sep)[-1]
        id = int(file_name[:-4].split('_')[-1])
        id_list.append(id)
    id_list.sort()
    
    mean_dist_total = 0.0
    for id in id_list:
        whether_valid, mean_dist = process_one_picture(base_dir, scene_id, id, save_dir)
        if whether_valid == True:
            valid_id_list.append(id)
            mean_dist_total += mean_dist
    mean_dist = mean_dist_total/ len(valid_id_list)
    print('mean_dist:', '{:.4f}'.format(mean_dist))

    valid_file_name = os.path.join(base_dir, scene_id, 'vp_list.txt')
    f = open(valid_file_name, 'w')
    for i in range(len(valid_id_list)):
        id = valid_id_list[i]
        f.write(str(id))
        if i != len(valid_id_list) - 1:
            f.write('\n')
    f.close()
    
def process_all(base_dir, save_dir, start_id):
    """Processing all data

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        save_dir [string]: [the save directory]
        start_id [string]: [the start processing id]
    """
    scene_list = []
    for name in glob.glob(os.path.join(base_dir, '*')):
        scene_id = name.split(os.sep)[-1]
        if scene_id == 'train.txt' or scene_id == 'valid.txt':
            continue 
        scene_list.append(scene_id)
    scene_list.sort()
    for scene_id in scene_list:
        if scene_id >= start_id:
            process_one_scene(base_dir, scene_id, save_dir)
        
def main():
    """The main function
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir', default='/home2/sgl/scannet_mine', type=str)
    parser.add_argument('--save_dir', default='/home2/sgl/data_test', type=str)
    parser.add_argument('--start_id', default='scene0000_01', type=str)
    args = parser.parse_args()
    process_all(args.base_dir, args.save_dir, args.start_id)

if __name__ == "__main__":
    main()
    #process_one_scene('/home2/sgl/scannet_mine', 'scene0000_01', '/home2/sgl/data_test')
            