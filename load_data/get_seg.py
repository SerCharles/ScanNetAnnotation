"""get the segmentation label based on label data provided by ScanNet
label 1 is background, 0 is others
"""

import numpy as np
import os
import argparse
import glob
import PIL.Image as Image

def load_grey_image(file_name):
    """Load a grey image(depth/norm/segmentation)
        H: the height of the picture
        W: the width of the picture

    Args:
        file_name [string]: [the full place of the image]

    Returns:
        pic_array [numpy float array], [H * W]: [the numpy data of the image]
    """
    fp = open(file_name, 'rb')
    pic = Image.open(fp)
    pic_array = np.array(pic)
    fp.close()
    return pic_array

def get_one_scene(base_dir, scene_id):
    """Get the segmentation label of one scene

    Args:
        base_dir [string] [the base directory of our modified scannet dataset]
        scene_id [string]: [scene id name]
    """

    save_dir = os.path.join(base_dir, scene_id, 'seg')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    full_name_list = glob.glob(os.path.join(base_dir, scene_id, 'label', "*.png"))
    id_list = []
    for full_name in full_name_list:
        file_name = full_name.split(os.sep)[-1]
        id = int(file_name[:-4].split('_')[-1])
        id_list.append(id)

    for id in id_list:
        source_name = os.path.join(base_dir, scene_id, 'label', scene_id + '_' + str(id) + '.png')
        target_name = os.path.join(base_dir, scene_id, 'seg', scene_id + '_' + str(id) + '.png')
        source_array = load_grey_image(source_name)
        target_array = (source_array == 1) | (source_array == 3) | (source_array == 41)
        target_array = (target_array).astype(np.uint16)
        picture = Image.fromarray(target_array)
        picture.save(target_name)
        print('written', target_name)

def main():
    """The main function of segmentation generation
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir', default='/home1/sgl/scannet_mine', type=str)
    args = parser.parse_args()

    for name in glob.glob(os.path.join(args.base_dir, '*')):
        scene_id = name.split(os.sep)[-1]
        if scene_id == 'train.txt' or scene_id == 'valid.txt':
            continue 
        get_one_scene(args.base_dir, scene_id)

if __name__ == "__main__":
    main()