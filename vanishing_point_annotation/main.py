"""The main function of vanishing point annotation
"""
import argparse
import os
import glob
import numpy as np
import utils
import data_utils
import time

def annotation_one_picture(base_dir, save_dir, scene_id, id):
    """Annotate one picture

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        save_dir [string]: [the base directory to save our result]
        scene_id [string]: [the scene id to be handled]
        id [int]: [the id of the picture]
    """

    intrinsic, extrinsic, layout_seg = data_utils.load_data(base_dir, scene_id, id)
    H, W = layout_seg.shape
    start = time.time()
    vy, vx = utils.get_vanishing_point(H, W, intrinsic, extrinsic)
    print(vy, vx)
    lines = utils.get_lines(H, W, vy, vx)
    whether_ceilings, whether_floors, whether_walls, ceiling_places, floor_places = utils.get_ceiling_and_floor(layout_seg, lines, 7, 4)
    whether_boundaries = utils.get_wall_boundaries(layout_seg, lines, 7, 4)
    
    end = time.time()
    print(end - start)
    full_save_dir = os.path.join(save_dir, scene_id + "_" + str(id) + '.png')
    data_utils.visualize_annotation_result(full_save_dir, layout_seg, lines, whether_ceilings, whether_floors, whether_walls, whether_boundaries, ceiling_places, floor_places)
    return end - start

def annotation_one_scene(base_dir, scene_id):
    """Annotate one scene

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the scene id to be handled]
    """
    save_dir = os.path.join(base_dir, scene_id, 'test')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    full_name_list = glob.glob(os.path.join(base_dir, scene_id, 'pose', "*.txt"))
    id_list = []
    for full_name in full_name_list:
        file_name = full_name.split(os.sep)[-1]
        id = int(file_name[:-4].split('_')[-1])
        id_list.append(id)
        
    total_time = 0.0
    for id in id_list:
        dtime = annotation_one_picture(base_dir, save_dir, scene_id, id)
        total_time += dtime 
    avg_time = total_time / len(id_list)
    print('total time is', total_time, 's')
    print('average time is', avg_time, 's')

def main():
    """The main function of vanishing point annotation
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir_scannet', default='/home1/sgl/scannet_mine', type=str)
    parser.add_argument('--base_dir_plane', default='/home1/shenguanlin/scannet_planes', type=str)
    parser.add_argument('--scene_id', default='scene0000_01', type=str)
    args = parser.parse_args()

    print('Rendering', args.scene_id)
    annotation_one_scene(args.base_dir_scannet, args.scene_id)
    

if __name__ == "__main__":
    main()