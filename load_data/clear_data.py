"""Scan the entire of our modified ScanNet dataset, deleting invalid and useless data
"""
import argparse
import os
import glob

def clear_data(base_dir, base_dir_plane, base_dir_original):
    """Split the training scenes and validation scenes

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        base_dir_plane [string]: [the base directory of our modified ScanNet-Planes dataset]
        base_dir_original [string]: [the base directory of our original ScanNet dataset]
    """
    scene_ids = []
    for name in glob.glob(os.path.join(base_dir, '*')):
        scene_id = name.split(os.sep)[-1]
        if scene_id == 'train.txt' or scene_id == 'valid.txt':
            continue
        scene_ids.append(scene_id)
    scene_ids.sort()
    count = 0
    for scene_id in scene_ids:        
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
        
        os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'ply'))
        os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'label'))
        os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'instance'))
        os.system('mkdir ' + os.path.join(base_dir, scene_id, 'ply'))
        os.system('cp ' + os.path.join(base_dir_plane, scene_id + '.ply') + ' ' + os.path.join(base_dir, scene_id, 'ply', scene_id + '_layout.ply'))
        os.system('cp ' + os.path.join(base_dir_original, scene_id, scene_id + '_vh_clean_2.ply') + ' ' + os.path.join(base_dir, scene_id, 'ply', scene_id + '.ply'))
        os.system('sudo chmod 777 ' + os.path.join(base_dir, scene_id, 'ply', scene_id + '.ply'))
        
        if num_depth <= 0 or num_seg != num_depth or \
            num_layout_seg != num_depth or num_layout_depth != num_depth or num_layout_norm / num_depth != 3 \
                or not(os.path.exists(os.path.join(base_dir_plane, scene_id + '.ply'))):
            
            count += 1
            if num_depth <= 0 or num_seg != num_depth:
                print(scene_id, 'has no basic')
                
                os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'depth'))
                os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'norm'))
            else: 
                print(scene_id, 'has no layout')
                os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'layout_depth'))
                os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'layout_norm'))
                os.system('rm -rf ' + os.path.join(base_dir, scene_id, 'layout_seg'))
    print(count)

def main():
    """The main function of data clearing
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir', default='/home1/sgl/scannet_mine', type=str)
    parser.add_argument('--base_dir_plane', default='/home1/sgl/scannet_planes_mine', type=str)
    parser.add_argument('--base_dir_original', default='/data/sgl/scannet/scans', type=str)

    args = parser.parse_args()
    clear_data(args.base_dir, args.base_dir_plane, args.base_dir_original)

if __name__ == "__main__":
    main()