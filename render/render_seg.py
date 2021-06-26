'''
render the layout info of ScanNet, including layout depth, norm, seg
'''

import sys
import numpy as np
import sys
import os
import argparse
import glob
import lib.render as render
from  data_loader import *
import time
from plyfile import *
import PIL.Image as Image


def render_one_scene(base_dir, scene_id):
    """[Render the seg of one scene]

    Args:
        base_dir ([str]): [the base directory of scannet]
        scene_id ([str]): [scene id name]
    """

    save_dir = os.path.join(base_dir, scene_id, 'seg')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    
    full_name_list = glob.glob(os.path.join(base_dir, scene_id, 'pose', "*.txt"))
    id_list = []
    for full_name in full_name_list:
        file_name = full_name.split(os.sep)[-1]
        id = int(file_name[:-4].split('_')[-1])
        id_list.append(id)

    intrinsic_name = os.path.join(base_dir, scene_id, '_info.txt')
    width, height, fx, fy, cx, cy = load_intrinsic(intrinsic_name)

    input_name = os.path.join(base_dir, scene_id, 'ply', scene_id + '_vh_clean_2.labels.ply')
    V, F = load_simple_ply(input_name)
    face_labels = get_seg_label(base_dir, scene_id, V, F)

    context = render.SetMesh(V, F)
    info = {'Height': height, 'Width': width, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    render.setup(info)

    for id in id_list:
        pose_name = scene_id + '_' + str(id) + '.txt'
        full_pose_name = os.path.join(base_dir, scene_id, 'pose', pose_name)
        pose = load_pose(full_pose_name)

        cam2world = pose 
        world2cam = np.linalg.inv(cam2world).astype('float32')
        render.render(context, world2cam)
        vindices, vweights, findices = render.getVMap(context, info)

        
        x_shape = findices.shape[0]
        y_shape = findices.shape[1]

        H = vindices.shape[0]
        W = vindices.shape[1]
        color_white = np.ones((H, W), dtype = np.uint32)

        face_labels = np.array(face_labels, dtype = int)
        final_labels = face_labels[findices]
        final_color = color_white * final_labels

        final_color = (final_color * 60000).astype(np.uint16)
        result_name = scene_id + '_' + str(id) + '.png'  
        full_result_name = os.path.join(base_dir, scene_id, 'seg', result_name)
        picture = Image.fromarray(final_color)
        picture.save(full_result_name)
        print('written', full_result_name)





def main():
    '''
    description: the main function of data rendering
    parameter: empty
    return: empty
    '''
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--base_dir', default = '/home1/shenguanlin/scannet_pretrain', type = str)
    parser.add_argument('--scene_id', default = 'scene0000_01', type = str)
    args = parser.parse_args()

    print('Rendering', args.scene_id)
    render_one_scene(args.base_dir, args.scene_id)

    

if __name__ == "__main__":
    main()