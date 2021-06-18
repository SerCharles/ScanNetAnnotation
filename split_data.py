'''
Used in getting the train and valid datasets
'''
import argparse
import numpy as np
import os
import glob

def split_train_valid(base_dir):
    """[split the training scenes and validation scenes]
    Args:
        base_dir ([str]): [the base directory of my handled scannet dataset]
    """
    train_name = os.path.join(base_dir, 'train.txt')
    valid_name = os.path.join(base_dir, 'valid.txt')
    f_train = open(train_name, 'w')
    f_valid = open(valid_name, 'w')
    for name in glob.glob(os.path.join(base_dir, '*')):
        scene_id = name.split(os.sep)[-1]
        if scene_id == 'train.txt' or scene_id == 'valid.txt':
            continue 
        
        try:
            depth_names = glob.glob(os.path.join(base_dir, scene_id, 'new_depth', '*.png'))
            norm_names = glob.glob(os.path.join(base_dir, scene_id, 'norm', '*.png'))
            num_depth = len(depth_names)
            num_norm = len(norm_names)
        except:
            num_depth = 0
            num_norm = 0

        if num_norm <= 0 or num_depth <= 0 or num_norm / num_depth != 3:
            print(scene_id)
            continue

        id_num = int(scene_id[5:9])
        if id_num <= 650:
            f_train.write(scene_id + '\n')
        else: 
            f_valid.write(scene_id + '\n')

    f_train.close()
    f_valid.close()



def main():
    """[main function of data clearing]
    """
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--base_dir', default = '/home1/shenguanlin/scannet_pretrain', type = str)
    args = parser.parse_args()
    split_train_valid(args.base_dir)

if __name__ == "__main__":
    main()