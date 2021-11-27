"""The main function of vanishing point annotation
"""
import argparse
import os
import json
import glob
import numpy as np
import utils
import data_utils
import time
import random 

def annotation_one_picture(base_dir, save_dir, scene_id, id, ceiling_id, floor_id):
    """Annotate one picture

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        save_dir [string]: [the base directory to save our result]
        scene_id [string]: [the scene id to be handled]
        id [int]: [the id of the picture]
        ceiling_id [int]: [the id of the plane which is the ceiling]
        floor_id [int]: [the id of the plane which is the floor]
    """
    intrinsic, extrinsic, layout_seg = data_utils.load_data(base_dir, scene_id, id)
    H, W = layout_seg.shape
    start = time.time()
    vanishing_point = utils.get_vanishing_point(H, W, intrinsic, extrinsic)
    vy = float(vanishing_point[0])
    if vy > H - 1 or vy < 0:
        boundary_angles, boundary_segs = utils.get_wall_boundaries(layout_seg, vanishing_point, ceiling_id, floor_id)
        wall_segs = utils.get_wall_seg(layout_seg, ceiling_id, floor_id)
        data_utils.save_boundaries(base_dir, scene_id, id, vanishing_point, boundary_angles, boundary_segs, wall_segs)
    end = time.time()
    return end - start

def annotation_one_scene(base_dir_scannet, base_dir_plane, scene_id):
    """Annotate one scene, 100 once a time

    Args:
        base_dir_scannet [string]: [the base directory of our modified ScanNet dataset]
        base_dir_plane [string]: [the base directory of our modified ScanNet-Planes dataset]
        scene_id [string]: [the scene id to be handled]
        start_num [int]: [the number of start]
        number_per_round [int]: [the number to be rendered per round]
    """
    json_place = os.path.join(base_dir_plane, scene_id + '.json')
    with open(json_place, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    ceiling_id = int(json_data['ceiling_id'])
    floor_id = int(json_data['floor_id'])
    
    save_dir = os.path.join(base_dir_scannet, scene_id, 'vanishing_point')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    full_name_list = glob.glob(os.path.join(base_dir_scannet, scene_id, 'pose', "*.txt"))
    full_name_list.sort()
    id_list = []
    for full_name in full_name_list:
        file_name = full_name.split(os.sep)[-1]
        id = int(file_name[:-4].split('_')[-1])
        id_list.append(id)
    id_list.sort()
    total_time = 0.0
    for i in range(len(id_list)):
        id = id_list[i]
        dtime = annotation_one_picture(base_dir_scannet, save_dir, scene_id, id, ceiling_id, floor_id)
        total_time += dtime 
    avg_time = total_time / len(id_list)
    print('Rendered', scene_id)
    print('total time is', total_time, 's')
    print('average time is', avg_time, 's')
    



def annotation_all(base_dir_scannet, base_dir_plane):
    """Processing all data

    Args:
        base_dir_scannet [string]: [the base directory of our modified ScanNet dataset]
        base_dir_plane [string]: [the base directory of our modified ScanNet-Planes dataset]
    """
    scene_list = []
    for name in glob.glob(os.path.join(base_dir_scannet, '*')):
        scene_id = name.split(os.sep)[-1]
        if scene_id == 'train.txt' or scene_id == 'valid.txt':
            continue 
        scene_list.append(scene_id)
    scene_list.sort()
    
    for scene_id in scene_list:
        annotation_one_scene(base_dir_scannet, base_dir_plane, scene_id)



def main():
    """The main function of vanishing point annotation
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir_scannet', default='/home1/shenguanlin/scannet_mine', type=str)
    parser.add_argument('--base_dir_plane', default='/home1/shenguanlin/scannet_planes_mine', type=str)
    args = parser.parse_args()
    annotation_all(args.base_dir_scannet, args.base_dir_plane)
    


    

if __name__ == "__main__":
    main()