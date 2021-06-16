'''
render the layout info of ScanNet, including layout depth, norm, seg
'''

from render.data_loader import load_planes
import sys
import numpy as np
import sys
import os
import argparse
import glob
import lib.render as render
from data_loader import *
import time
import PIL.Image as Image


def render_one_scene(base_dir_scannet, base_dir_plane, scene_id):
    """[Render the layout norm, seg, depth of one scene]

    Args:
        base_dir_scannet ([str]): [the base directory of scannet]
        base_dir_plane ([str]): [the base directory of scannet-plane]
        scene_id ([str]): [scene id name]
    """
    save_dir = os.path.join(base_dir_scannet, scene_id, 'layout_norm')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(base_dir_scannet, scene_id, 'layout_depth')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(base_dir_scannet, scene_id, 'layout_seg')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    
    full_name_list = glob.glob(os.path.join(base_dir_scannet, scene_id, 'pose', "*.txt"))
    id_list = []
    for full_name in full_name_list:
        file_name = full_name.split(os.sep)[-1]
        id = int(file_name[:-4].split('_')[-1])
        id_list.append(id)

    intrinsic_name = os.path.join(base_dir_scannet, scene_id, '_info.txt')
    width, height, fx, fy, cx, cy = load_intrinsic(intrinsic_name)

    V, F, NORM, L = load_planes(base_dir_plane, scene_id)
    context = render.SetMesh(V, F)
    info = {'Height': height, 'Width': width, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    render.setup(info)

    for id in id_list:
        pose_name = scene_id + '_' + str(id) + '.txt'
        full_pose_name = os.path.join(base_dir_scannet, scene_id, 'pose', pose_name)
        pose = load_pose(full_pose_name)

        cam2world = pose 
        world2cam = np.linalg.inv(cam2world).astype('float32')
        render.render(context, world2cam)
        vindices, vweights, findices = render.getVMap(context, info)
        depth = render.getDepth(info)

        
        the_norm = transform_norm(NORM, world2cam)
        x_shape = findices.shape[0]
        y_shape = findices.shape[1]

        final_seg = np.zeros((x_shape, y_shape), dtype = 'int32')
        final_norm = np.zeros((x_shape, y_shape, 3), dtype ='float32')
        H = vindices.shape[0]
        W = vindices.shape[1]

        final_seg = L[findices]
        final_norm = the_norm[findices]
        print(final_seg.shape)
        print(final_norm.shape)

    
        final_depth = (depth * 1000).astype(np.uint16)
        final_norm = (final_norm * 32768).astype(np.uint16)
        final_seg = (final_seg * 3000).astype(np.uint16)

        result_name_seg = scene_id + '_' + str(id) + '.png'  
        result_name_depth = scene_id + '_' + str(id) + '.png'  
        result_name_nx = scene_id + '_' + str(id) + '_nx.png'  
        result_name_ny = scene_id + '_' + str(id) + '_ny.png'  
        result_name_nz = scene_id + '_' + str(id) + '_nz.png'  

        full_result_name_seg = os.path.join(base_dir_scannet, scene_id, 'layout_seg', result_name_seg)
        full_result_name_depth = os.path.join(base_dir_scannet, scene_id, 'layout_depth', result_name_depth)
        full_result_name_nx = os.path.join(base_dir_scannet, scene_id, 'layout_norm', result_name_nx)
        full_result_name_ny = os.path.join(base_dir_scannet, scene_id, 'layout_norm', result_name_ny)
        full_result_name_nz = os.path.join(base_dir_scannet, scene_id, 'layout_norm', result_name_nz)

        picture_seg = Image.fromarray(final_seg)
        picture_seg.save(full_result_name_seg)
        print('written', full_result_name_seg)

        picture_depth = Image.fromarray(final_depth)
        picture_depth.save(full_result_name_depth)
        print('written', full_result_name_depth)


        final_nx = final_norm[:, :, 0]
        picture_nx = Image.fromarray(final_nx)
        picture_nx.save(full_result_name_nx)
        print('written', full_result_name_nx)

        final_ny = final_norm[:, :, 1]
        picture_ny = Image.fromarray(final_ny)
        picture_ny.save(full_result_name_ny)
        print('written', full_result_name_ny)

        final_nz = final_norm[:, :, 2]
        picture_nz = Image.fromarray(final_nz)
        picture_nz.save(full_result_name_nz)
        print('written', full_result_name_nz)


        



def main():
    '''
    description: the main function of data rendering
    parameter: empty
    return: empty
    '''
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--base_dir_scannet', default = '/home1/shenguanlin/scannet_pretrain', type = str)
    parser.add_argument('--base_dir_plane', default = '/home1/shenguanlin/scannet_planes', type = str)
    parser.add_argument('--scene_id', default = 'scene0000_01', type = str)
    args = parser.parse_args()

    print('Rendering', args.scene_id)
    render_one_scene(args.base_dir_scannet, args.base_dir_plane, args.scene_id)

    

if __name__ == "__main__":
    main()