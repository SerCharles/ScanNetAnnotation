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

def read_intrinsic(full_name):
    '''
    description: read the intrinsic
    parameter: full_name
    return: fx, fy, cx, cy
    '''
    f = open(full_name, 'r')
    words = f.read().split()
    fx = float(words[0])
    fy = float(words[1])
    cx = float(words[2])
    cy = float(words[3])
    f.close()
    return fx, fy, cx, cy

def read_pose(full_name):
    '''
    description: read the extrinsic
    parameter: full_name
    return: numpy array of extrinsic
    '''
    pose = np.zeros((4, 4), dtype = np.float32)
    f = open(full_name, 'r')
    lines = f.read().split('\n')
    
    for i in range(4):
        words = lines[i].split()
        for j in range(4):
            word = float(words[j])
            pose[i][j] = word

    pose = pose.astype(np.float32)
    f.close()
    return pose

def render_one_scene(base_dir, scene_name, picture_name_list):
    '''
    description: render one scene
    parameter: the base dir of data, the scene name, the list of picture name lists
    return: empty
    '''
    #input_name = os.path.join(base_dir, 'mesh', scene_name + '_seg.ply')
    input_name = '/home/shenguanlin/scene0000_00_vh_clean.ply'
    start = time.time()
    V, F, NORM, C = LoadPLY(input_name)
    end = time.time()
    print(end - start, 's')

    context = render.SetMesh(V, F)

    for i in range(len(picture_name_list)):
        start = time.time()

        '''
        picture_name = picture_name_list[i]
        base_name = picture_name[:-9]
        group_name = picture_name[-7]
        ins_name = picture_name[-5]
        full_name = base_name + '_pose_' + group_name + '_' + ins_name + '.txt'

        full_name_intrinsic = os.path.join(base_dir, 'intrinsic', full_name)
        full_name_pose = os.path.join(base_dir, 'pose', full_name)
        fx, fy, cx, cy = read_intrinsic(full_name_intrinsic)

        pose = read_pose(full_name_pose)
        '''
        pose = [[-0.955421, 0.119616, -0.269932, 2.655830],
            [0.295248, 0.388339, -0.872939, 2.981598],
            [0.000408, -0.913720, -0.406343, 1.368648],
            [0.000000, 0.000000, 0.000000, 1.000000]]

        info = {'Height':968, 'Width':1296, 'fx':1170.187988, 'fy':1170.187988, 'cx':647.75, 'cy':483.75}
        render.setup(info)

        cam2world = pose 
        world2cam = np.linalg.inv(cam2world).astype('float32')
        render.render(context, world2cam)
        vindices, vweights, findices = render.getVMap(context, info)
        #render.Clear()
        the_norm = transform_norm(NORM, world2cam)

        x_shape = findices.shape[0]
        y_shape = findices.shape[1]
        final_color = np.zeros((x_shape, y_shape, 3), dtype ='float32')
        final_norm = np.zeros((x_shape, y_shape, 3), dtype ='float32')


        H = vindices.shape[0]
        W = vindices.shape[1]

        for k in range(vindices.shape[2]):
            indice = vindices[:, :, k]
            weight = vweights[:, :, k]
            weight = np.reshape(weight, (H, W, 1))
            weight = np.repeat(weight, 3, axis = 2)
            norm_value = the_norm[indice]
            color_value = C[indice]
            final_color = final_color + color_value * weight
            final_norm = final_norm + norm_value * weight

        final_norm = (final_norm * 32768).astype(np.uint16)
        final_color = (final_color * 256).astype(np.uint8)
        #result_name = base_name + '_s' + group_name + '_' + ins_name + '.png'
        #full_result_name = os.path.join(base_dir, 'segs', result_name)
        full_result_name = '/home/shenguanlin/kebab.png'
        full_result_name_nx = '/home/shenguanlin/nx.png'
        full_result_name_ny = '/home/shenguanlin/ny.png'
        full_result_name_nz = '/home/shenguanlin/nz.png'
        
        picture = Image.fromarray(final_color, mode = 'RGB')
        picture.save(full_result_name)
        
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
        

        end = time.time()
        print(end - start, 's')


def get_scene_names(base_dir, scene_name):
    '''
    description: get the names of the relative picture names of the scene
    parameter: the base dir of data
    return: name lists
    '''
    picture_name_list = []
    file_name = os.path.join(base_dir, 'data_list', scene_name + '.conf')
    f = open(file_name, 'r')
    lines = f.read().split('\n')
    for line in lines:
        words = line.split()
        if len(words) > 0 and words[0] == 'scan':
            picture_name_list.append(words[1])
    f.close()
    return picture_name_list


def main():
    '''
    description: the main function of data rendering
    parameter: empty
    return: empty
    '''
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--base_dir', default = '/data/sgl/geolayout_pretrain', type = str)
    parser.add_argument('--conf_name', default = 'JF19kD82Mey.conf', type = str)
    args = parser.parse_args()
    args.scene_name = args.conf_name[:-5]

    #picture_name_list = get_scene_names(args.base_dir, args.scene_name)
    #print('Rendering', args.scene_name)
    picture_name_list = ['kebab']
    render_one_scene(args.base_dir, args.scene_name, picture_name_list)

    

if __name__ == "__main__":
    main()