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
import lib.cuda_caller

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
    vy, vx = utils.get_vanishing_point(H, W, intrinsic, extrinsic)
    lines = utils.get_lines(H, W, vy, vx)
    whether_ceilings, whether_floors, whether_walls, ceiling_places, floor_places = \
        lib.cuda_caller.get_ceiling_and_floor(layout_seg, lines, ceiling_id, floor_id)
    whether_boundaries = utils.get_wall_boundaries(layout_seg, lines, ceiling_id, floor_id)
    
    end = time.time()
    full_save_dir = os.path.join(save_dir, scene_id + "_" + str(id) + '.npz')
    #data_utils.visualize_annotation_result(full_save_dir, layout_seg, lines, whether_ceilings, whether_floors, whether_walls, whether_boundaries, ceiling_places, floor_places)
    data_utils.save_annotation_result(full_save_dir, vy, vx, whether_ceilings, whether_floors, whether_walls, whether_boundaries, ceiling_places, floor_places)
    return end - start

def annotation_one_scene(base_dir_scannet, base_dir_plane, scene_id):
    """Annotate one scene

    Args:
        base_dir_scannet [string]: [the base directory of our modified ScanNet dataset]
        base_dir_plane [string]: [the base directory of our modified ScanNet-Planes dataset]
        scene_id [string]: [the scene id to be handled]
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
    id_list = []
    for full_name in full_name_list:
        file_name = full_name.split(os.sep)[-1]
        id = int(file_name[:-4].split('_')[-1])
        id_list.append(id)
        
    total_time = 0.0
    for id in id_list:
        dtime = annotation_one_picture(base_dir_scannet, save_dir, scene_id, id, ceiling_id, floor_id)
        total_time += dtime 
    avg_time = total_time / len(id_list)
    print('total time is', total_time, 's')
    print('average time is', avg_time, 's')

def main():
    """The main function of vanishing point annotation
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir_scannet', default='/home1/sgl/scannet_mine', type=str)
    parser.add_argument('--base_dir_plane', default='/home1/sgl/scannet_planes_mine', type=str)
    parser.add_argument('--scene_id', default='scene0000_01', type=str)
    args = parser.parse_args()

    print('Rendering', args.scene_id)
    annotation_one_scene(args.base_dir_scannet, args.base_dir_plane, args.scene_id)
    

if __name__ == "__main__":
    main()