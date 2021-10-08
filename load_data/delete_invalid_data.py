"""Delete the invalid data
"""
import argparse
import os
import glob
import numpy as np

def load_extrinsic(file_name):
    """load an extrinsic file of ScanNet

    Args:
        file_name [str]: [the name of the extrinsic file]

    Returns:
        pose [numpy float array], [4 * 4]: [the extrinsic numpy array]
    """
    pose = np.zeros((4, 4), dtype = np.float32)
    f = open(file_name, 'r')
    lines = f.read().split('\n')
    
    for i in range(4):
        words = lines[i].split()
        for j in range(4):
            word = float(words[j])
            pose[i][j] = word

    pose = pose.astype(np.float32)
    f.close()
    return pose

def delete_one_group(base_dir, scene_id, id):
    """Delete the data of one group

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the scene id]
        id [string]: [the id of the picture]
    """
    #get name
    base_name = scene_id + '_' + id
    extrinsic_name = scene_id + "_" + id + '.txt'
    image_name = scene_id + '_' + id + '.jpg'
    depth_name = scene_id + '_' + id + '.png'
    nx_name = scene_id + '_' + id + '_nx.png'
    ny_name = scene_id + '_' + id + '_ny.png'
    nz_name = scene_id + '_' + id + '_nz.png'
    seg_name = scene_id + '_' + id + '.png'
    layout_depth_name = scene_id + '_' + id + '.png'
    layout_nx_name = scene_id + '_' + id + '_nx.png'
    layout_ny_name = scene_id + '_' + id + '_ny.png'
    layout_nz_name = scene_id + '_' + id + '_nz.png'
    layout_seg_name = scene_id + '_' + id + '.png'
    vp_name = scene_id + '_' + id + '.npz'

    #get full name
    extrinsic_full_name = os.path.join(base_dir, scene_id, 'pose', extrinsic_name)
    image_full_name = os.path.join(base_dir, scene_id, 'color', image_name)
    depth_full_name = os.path.join(base_dir, scene_id, 'depth', depth_name)
    nx_full_name = os.path.join(base_dir, scene_id, 'norm', nx_name)
    ny_full_name = os.path.join(base_dir, scene_id, 'norm', ny_name)
    nz_full_name = os.path.join(base_dir, scene_id, 'norm', nz_name)
    seg_full_name = os.path.join(base_dir, scene_id, 'seg', seg_name)
    layout_depth_full_name = os.path.join(base_dir, scene_id, 'layout_depth', layout_depth_name)
    layout_nx_full_name = os.path.join(base_dir, scene_id, 'layout_norm', layout_nx_name)
    layout_ny_full_name = os.path.join(base_dir, scene_id, 'layout_norm', layout_ny_name)
    layout_nz_full_name = os.path.join(base_dir, scene_id, 'layout_norm', layout_nz_name)
    layout_seg_full_name = os.path.join(base_dir, scene_id, 'layout_seg', layout_seg_name) 
    vp_full_name = os.path.join(base_dir, scene_id, 'vanishing_point', vp_name) 

    extrinsic = load_extrinsic(extrinsic_full_name)
    valid = True 
    for i in range(4):
        for j in range(4):
            try: 
                kebab = int(extrinsic[i][j])
            except:
                valid = False 
    if valid == False: 
        os.remove(extrinsic_full_name)
        os.remove(image_full_name)
        os.remove(depth_full_name)
        os.remove(nx_full_name)
        os.remove(ny_full_name)
        os.remove(nz_full_name)
        os.remove(seg_full_name)
        os.remove(layout_depth_full_name)
        os.remove(layout_nx_full_name)
        os.remove(layout_ny_full_name)
        os.remove(layout_nz_full_name)
        os.remove(layout_seg_full_name)
        os.remove(vp_full_name)
        print('Removed', base_name)

def delete_invalid_data(base_dir):
    """Delete the invalid data

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
    """
    for name in glob.glob(os.path.join(base_dir, '*')):
        scene_id = name.split(os.sep)[-1]
        if scene_id == 'train.txt' or scene_id == 'valid.txt':
            continue 
        
        for full_id in glob.glob(os.path.join(base_dir, scene_id, 'pose', '*.txt')):
            file_name = full_id.split(os.sep)[-1]
            id = int(file_name[:-4].split('_')[-1])
            delete_one_group(base_dir, scene_id, str(id))




def main():
    """The main function of data clearing
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir', default='/home1/sgl/scannet_mine', type=str)
    args = parser.parse_args()
    delete_invalid_data(args.base_dir)

if __name__ == "__main__":
    main()