"""Scan the entire of our modified ScanNet dataset, deleting invalid and useless data, and split train/valid scenes
"""
import argparse
import os
import glob

def split_train_valid(base_dir, base_dir_plane, whether_clear):
    """Split the training scenes and validation scenes

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        base_dir_plane [string]: [the base directory of our modified ScanNet-Planes dataset]
        whether_clear [boolean]: [clear useless data or not]
    """
    train_name = os.path.join(base_dir, 'train.txt')
    valid_name = os.path.join(base_dir, 'valid.txt')
    train_list = []
    valid_list = []
    f_train = open(train_name, 'w')
    f_valid = open(valid_name, 'w')
    for name in glob.glob(os.path.join(base_dir, '*')):
        scene_id = name.split(os.sep)[-1]
        if scene_id == 'train.txt' or scene_id == 'valid.txt':
            continue 
        
        try:
            depth_names = glob.glob(os.path.join(base_dir, scene_id, 'depth', '*.png'))
            norm_names = glob.glob(os.path.join(base_dir, scene_id, 'norm', '*.png'))
            seg_names = glob.glob(os.path.join(base_dir, scene_id, 'seg', '*.png'))
            layout_seg_names = glob.glob(os.path.join(base_dir, scene_id, 'layout_seg', '*.png'))
            layout_norm_names = glob.glob(os.path.join(base_dir, scene_id, 'layout_norm', '*.png'))
            layout_depth_names = glob.glob(os.path.join(base_dir, scene_id, 'layout_depth', '*.png'))

            num_depth = len(depth_names)
            num_norm = len(norm_names)
            num_seg = len(seg_names)
            num_layout_seg = len(layout_seg_names)
            num_layout_norm = len(layout_norm_names)
            num_layout_depth = len(layout_depth_names)

        except:
            num_depth = 0
            num_norm = 0
            num_seg = 0
            num_layout_seg = 0
            num_layout_norm = 0
            num_layout_depth = 0

        
        if num_depth <= 0 or num_norm / num_depth != 3 or num_seg != num_depth or \
            num_layout_seg != num_depth or num_layout_depth != num_depth or num_layout_norm / num_depth != 3 or \
            (not os.path.exists(os.path.join(base_dir_plane, scene_id + '.ply'))) or \
                (not os.path.exists(os.path.join(base_dir_plane, scene_id + '_full.ply'))):
            print(scene_id, 'is invalid')

            if whether_clear != 0:
                try:
                    os.system('rm -rf ' + os.path.join(base_dir, scene_id))
                    os.remove(os.path.join(base_dir_plane, scene_id + '.ply'))
                    os.remove(os.path.join(base_dir_plane, scene_id + '_full.ply'))
                    os.remove(os.path.join(base_dir_plane, scene_id + '.json'))
                except: 
                    pass
            continue


        id_num = int(scene_id[5:9])
        if id_num <= 650:
            train_list.append(scene_id)
        else: 
            valid_list.append(scene_id)

        if whether_clear != 0:
            try:
                os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'ply'))
                os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'label'))
                os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'instance'))
                os.system('mkdir ' + os.path.join(base_dir, scene_id, 'ply'))
                os.system('cp ' + os.path.join(base_dir_plane, scene_id + '.ply') + ' ' + os.path.join(base_dir, scene_id, 'ply', scene_id + '_mesh.ply'))
                os.system('cp ' + os.path.join(base_dir_plane, scene_id + '_full.ply') + ' ' + os.path.join(base_dir, scene_id, 'ply', scene_id + '_pcd.ply'))
            except: 
                pass

    train_list.sort()
    valid_list.sort()
    for scene_id in train_list:
        f_train.write(scene_id + '\n')
    for scene_id in valid_list:
        f_valid.write(scene_id + '\n')

    f_train.close()
    f_valid.close()



def main():
    """The main function of data clearing
    """
    parser = argparse.ArgumentParser(descriptio ='')
    parser.add_argument('--base_dir', default='/home1/shenguanlin/scannet_mine', type=str)
    parser.add_argument('--base_dir_plane', default='/home1/shenguanlin/scannet_planes', type=str)
    parser.add_argument('--clear', default=0, type=int)

    args = parser.parse_args()
    split_train_valid(args.base_dir, args.base_dir_plane, args.clear)

if __name__ == "__main__":
    main()