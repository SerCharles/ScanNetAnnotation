'''
Used in clearing the data of the scannet dataset, leaving only 1/10
'''
import argparse
import numpy as np
import os
import glob

def save_remove(name):
    '''
    description: remove a file safely
    parameter: file name
    return: empty
    '''
    try: 
        os.remove(name)
        print('removed', name)
    except: 
        pass

def save_rename(source, target):
    '''
    description: rename a file safely
    parameter: source, target
    return: empty
    '''
    try: 
        os.rename(source, target)
        print('renamed', source)
    except: 
        pass

def save_move(source, target):
    '''
    description: move a file safely
    parameter: source, target
    return: empty
    '''
    try: 
        os.system('mv ' + source + ' ' +  target)
        print('moved', target)
    except: 
        pass

def save_mkdir(name):
    '''
    description: make a directory savely
    parameter: dir name
    return: empty
    '''
    if not os.path.exists(name):
        os.mkdir(name)

def clear_one(base_dir, scene_id):
    ''' 
    description: clearing the data in scannet dataset, leaving only 1/10 of the data
    parameter: the base dir of scannet data, the scene id
    return: empty
    '''
    save_mkdir(os.path.join(base_dir, scene_id, 'color'))
    save_mkdir(os.path.join(base_dir, scene_id, 'depth'))
    save_mkdir(os.path.join(base_dir, scene_id, 'pose'))
    save_mkdir(os.path.join(base_dir, scene_id, 'instance'))
    save_mkdir(os.path.join(base_dir, scene_id, 'label'))
    for name in glob.glob(os.path.join(base_dir, scene_id, 'instance-filt', '*.png')):
        print(name)
        filename = name.split(os.sep)[-1]
        print(filename)
        id = int(filename[:-4])
        color_name = os.path.join(base_dir, scene_id, 'frame-' + str(id).zfill(6) + '.color.jpg')
        depth_name = os.path.join(base_dir, scene_id, 'frame-' + str(id).zfill(6) + '.depth.pgm')
        pose_name = os.path.join(base_dir, scene_id, 'frame-' + str(id).zfill(6) + '.pose.txt')
        instance_name = os.path.join(base_dir, scene_id, 'instance-filt', str(id) + '.png')
        label_name = os.path.join(base_dir, scene_id, 'label-filt', str(id) + '.png')


        if id % 10 != 0:
            save_remove(color_name)
            save_remove(depth_name)
            save_remove(pose_name)
            save_remove(instance_name)
            save_remove(label_name)
        else: 

            new_name = scene_id + '_' + str(id)
            new_color_name = os.path.join(base_dir, scene_id, 'color', new_name + '.jpg')
            new_depth_name = os.path.join(base_dir, scene_id, 'depth', new_name + '.pgm')
            new_pose_name = os.path.join(base_dir, scene_id, 'pose', new_name + '.txt')
            new_instance_name = os.path.join(base_dir, scene_id, 'instance', new_name + '.png')
            new_label_name = os.path.join(base_dir, scene_id, 'label', new_name + '.png')
            save_move(color_name, new_color_name)
            save_move(depth_name, new_depth_name)
            save_move(pose_name, new_pose_name)
            save_move(instance_name, new_instance_name)
            save_move(label_name, new_label_name)
    os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'instance-filt'))
    os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'label-filt'))


def main():
    '''
    description: the main function of data clearing
    parameter: empty
    return: empty
    '''
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--base_dir', default = '/home/shenguanlin/scannet_pretrain', type = str)
    parser.add_argument('--scene_id', default = 'scene0000_00', type = str)
    args = parser.parse_args()

    clear_one(args.base_dir, args.scene_id)

    

if __name__ == "__main__":
    main()

