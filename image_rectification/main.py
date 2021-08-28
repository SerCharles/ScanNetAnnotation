"""Rectify the images
"""
import os
import glob
import argparse
import numpy as np
import torch
from math import *
import PIL.Image as Image
import data_utils
import utils

def load_one_group(base_dir, scene_id, picture_id):
    """Load one group of data

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the id of the scene]
        picture_id [int]: [the number of the frame of the picture group to be loaded]

    Returns:
        base_name [string]: [the base name of the data]
        extrinsic [numpy float array], [4 * 4]: [the extrinsic of the picture]
        intrinsic [numpy float array], [3 * 3]: [the intrinsic of the picture]
        image [numpy float array], [3 * H * W]: [the picture]
        seg [numpy boolean array], [1 * H * W]: [the segmentation masking whether the pixel is background or not, 1yes 0no]
        depth [numpy float array], [1 * H * W]: [the depth map of the picture]
        normal [numpy float array], [3 * H * W]: [the normal map of the picture]
        layout_seg [numpy int array], [1 * H * W]: [the background segmentation of the picture]
        layout_depth [numpy float array], [1 * H * W]: [the background depth map of the picture]
        layout_normal [numpy float array], [3 * H * W]: [the background normal map of the picture]
        mask [numpy boolean array], [1 * H * W]: [the segmentation masking whether the pixel is useful or not, 1yes 0no]
    """

    base_name = scene_id + '_' + str(picture_id)
    intrinsic_name = os.path.join(base_dir, scene_id, '_info.txt')
    extrinsic_name = os.path.join(base_dir, scene_id, 'pose', base_name + '.txt')
    color_name = os.path.join(base_dir, scene_id, 'color', base_name + '.jpg')
    depth_name = os.path.join(base_dir, scene_id, 'depth', base_name + '.png')
    nx_name = os.path.join(base_dir, scene_id, 'norm', base_name + '_nx.png')
    ny_name = os.path.join(base_dir, scene_id, 'norm', base_name + '_ny.png')
    nz_name = os.path.join(base_dir, scene_id, 'norm', base_name + '_nz.png')
    seg_name = os.path.join(base_dir, scene_id, 'seg', base_name + '.png')
    layout_depth_name = os.path.join(base_dir, scene_id, 'layout_depth', base_name + '.png')
    layout_nx_name = os.path.join(base_dir, scene_id, 'layout_norm', base_name + '_nx.png')
    layout_ny_name = os.path.join(base_dir, scene_id, 'layout_norm', base_name + '_ny.png')
    layout_nz_name = os.path.join(base_dir, scene_id, 'layout_norm', base_name + '_nz.png')
    layout_seg_name = os.path.join(base_dir, scene_id, 'layout_seg', base_name + '.png')
    
    intrinsic = data_utils.load_intrinsic(intrinsic_name)
    extrinsic = data_utils.load_extrinsic(extrinsic_name)
    image = data_utils.load_image(color_name) #H * W * 3
    H, W, _ = image.shape
    image = image.transpose(2, 0, 1).astype(np.float32) / 256.0 #3 * H * W
    depth = data_utils.load_image(depth_name).reshape(1, H, W).astype(np.float32) / 1000.0 #1 * H * W
    nx = data_utils.load_image(nx_name).reshape(1, H, W).astype(np.float32) / 32768.0 - 1.0 #1 * H * W
    ny = data_utils.load_image(ny_name).reshape(1, H, W).astype(np.float32) / 32768.0 - 1.0 #1 * H * W
    nz = data_utils.load_image(nz_name).reshape(1, H, W).astype(np.float32) / 32768.0 - 1.0 #1 * H * W
    normal = np.concatenate((nx, ny, nz), axis=0) #3 * H * W
    seg = data_utils.load_image(seg_name).reshape(1, H, W) #1 * H * W
    mask_original = data_utils.get_mask(depth, normal) #1 * H * W
    layout_depth = data_utils.load_image(layout_depth_name).reshape(1, H, W).astype(np.float32) / 1000.0 #1 * H * W
    layout_nx = data_utils.load_image(layout_nx_name).reshape(1, H, W).astype(np.float32) / 32768.0 - 1.0 #1 * H * W
    layout_ny = data_utils.load_image(layout_ny_name).reshape(1, H, W).astype(np.float32) / 32768.0 - 1.0 #1 * H * W
    layout_nz = data_utils.load_image(layout_nz_name).reshape(1, H, W).astype(np.float32) / 32768.0 - 1.0 #1 * H * W
    layout_normal = np.concatenate((layout_nx, layout_ny, layout_nz), axis=0) #3 * H * W
    layout_seg = data_utils.load_image(layout_seg_name).reshape(1, H, W) #1 * H * W
    mask_layout = data_utils.get_mask(layout_depth, layout_normal)
    mask = mask_original & mask_layout
    
    return base_name, extrinsic, intrinsic, image, seg, depth, normal, layout_seg, layout_depth, layout_normal, mask

def save_one_group(base_dir, scene_id, picture_id, base_name, extrinsic, image, seg, depth, normal, layout_seg, layout_depth, layout_normal, mask):
    """Save the group of data into the dataset

    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the id of the scene]
        picture_id [int]: [the number of the frame of the picture group to be saved]
        base_name [string]: [the base name of the data]
        extrinsic [numpy float array], [4 * 4]: [the extrinsic of the picture]
        image [numpy float array], [3 * H * W]: [the picture]
        seg [numpy boolean array], [1 * H * W]: [the segmentation masking whether the pixel is background or not, 1yes 0no]
        depth [numpy float array], [1 * H * W]: [the depth map of the picture]
        normal [numpy float array], [3 * H * W]: [the normal map of the picture]
        layout_seg [numpy int array], [1 * H * W]: [the background segmentation of the picture]
        layout_depth [numpy float array], [1 * H * W]: [the background depth map of the picture]
        layout_normal [numpy float array], [3 * H * W]: [the background normal map of the picture]
        mask [numpy boolean array], [1 * H * W]: [the segmentation masking whether the pixel is useful or not, 1yes 0no]
    """
    _, H, W = image.shape
    base_name = scene_id + '_' + str(picture_id)
    
    extrinsic_dir = os.path.join(base_dir, scene_id, 'rectified_pose')
    color_dir = os.path.join(base_dir, scene_id, 'rectified_color')
    depth_dir = os.path.join(base_dir, scene_id, 'rectified_depth')
    normal_dir = os.path.join(base_dir, scene_id, 'rectified_normal')
    seg_dir = os.path.join(base_dir, scene_id, 'rectified_seg')
    layout_depth_dir = os.path.join(base_dir, scene_id, 'rectified_layout_depth')
    layout_normal_dir = os.path.join(base_dir, scene_id, 'rectified_layout_normal')
    layout_seg_dir = os.path.join(base_dir, scene_id, 'rectified_layout_seg')
    mask_dir = os.path.join(base_dir, scene_id, 'rectified_mask')
    if not os.path.exists(extrinsic_dir):
        os.mkdir(extrinsic_dir)
    if not os.path.exists(color_dir):
        os.mkdir(color_dir)
    if not os.path.exists(depth_dir):
        os.mkdir(depth_dir)
    if not os.path.exists(normal_dir):
        os.mkdir(normal_dir)        
    if not os.path.exists(seg_dir):
        os.mkdir(seg_dir)        
    if not os.path.exists(layout_depth_dir):
        os.mkdir(layout_depth_dir)
    if not os.path.exists(layout_normal_dir):
        os.mkdir(layout_normal_dir)        
    if not os.path.exists(layout_seg_dir):
        os.mkdir(layout_seg_dir) 
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)  
              
    extrinsic_name = os.path.join(extrinsic_dir, base_name + '.txt')
    color_name = os.path.join(color_dir, base_name + '.jpg')
    depth_name = os.path.join(depth_dir, base_name + '.png')
    nx_name = os.path.join(normal_dir, base_name + '_nx.png')
    ny_name = os.path.join(normal_dir, base_name + '_ny.png')
    nz_name = os.path.join(normal_dir, base_name + '_nz.png')
    seg_name = os.path.join(seg_dir, base_name + '.png')
    layout_depth_name = os.path.join(layout_depth_dir, base_name + '.png')
    layout_nx_name = os.path.join(layout_normal_dir, base_name + '_nx.png')
    layout_ny_name = os.path.join(layout_normal_dir, base_name + '_ny.png')
    layout_nz_name = os.path.join(layout_normal_dir, base_name + '_nz.png')
    layout_seg_name = os.path.join(layout_seg_dir, base_name + '.png')
    mask_name = os.path.join(mask_dir, base_name + '.png')

    data_utils.save_extrinsic(extrinsic_name, extrinsic)
    color = Image.fromarray((image * 256).transpose(1, 2, 0).astype(np.uint8))
    color.save(color_name)
    depth = Image.fromarray((depth * 1000).reshape(H, W).astype(np.uint16))
    depth.save(depth_name)
    nx = Image.fromarray(((normal[0, :, :] + 1.0) * 32768).astype(np.uint16))
    nx.save(nx_name)
    ny = Image.fromarray(((normal[1, :, :] + 1.0) * 32768).astype(np.uint16))
    ny.save(ny_name)
    nz = Image.fromarray(((normal[2, :, :] + 1.0) * 32768).astype(np.uint16))
    nz.save(nz_name)
    seg = Image.fromarray((seg * 7000).reshape(H, W).astype(np.uint16))
    seg.save(seg_name)
    layout_depth = Image.fromarray((layout_depth * 1000).reshape(H, W).astype(np.uint16))
    layout_depth.save(layout_depth_name)
    layout_nx = Image.fromarray(((layout_normal[0, :, :] + 1.0) * 32768).astype(np.uint16))
    layout_nx.save(layout_nx_name)
    layout_ny = Image.fromarray(((layout_normal[1, :, :] + 1.0) * 32768).astype(np.uint16))
    layout_ny.save(layout_ny_name)
    layout_nz = Image.fromarray(((layout_normal[2, :, :] + 1.0) * 32768).astype(np.uint16))
    layout_nz.save(layout_nz_name)
    layout_seg = Image.fromarray((layout_seg.reshape(H, W) * 7000).astype(np.uint16))
    layout_seg.save(layout_seg_name)
    mask = Image.fromarray((mask * 60000).reshape(H, W).astype(np.uint16))
    mask.save(mask_name)
    print('Written', base_name)

def rectify_one_scene(cuda, base_dir, scene_id):
    """Rectify the pictures of one scene

    Args:
        cuda [boolean]: [Whether use GPU or not]
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the id of the scene]
    """
    print('Processing', scene_id)
    full_name_list = glob.glob(os.path.join(base_dir, scene_id, 'color', "*.jpg"))
    id_list = []
    for full_name in full_name_list:
        file_name = full_name.split(os.sep)[-1]
        id = int(file_name[:-4].split('_')[-1])
        id_list.append(id)

    total_rate = 0.0
    for id in id_list:
        base_name, extrinsic, intrinsic, image, seg, depth, normal, layout_seg, layout_depth, layout_normal, mask\
            = load_one_group(base_dir, scene_id, id)
        
        _, H, W = image.shape
        extrinsic = torch.from_numpy(extrinsic).view(1, 4, 4)
        intrinsic = torch.from_numpy(intrinsic).view(1, 3, 3)
        image = torch.from_numpy(image).view(1, 3, H, W)
        seg = torch.from_numpy(seg).view(1, 1, H, W)
        depth = torch.from_numpy(depth).view(1, 1, H, W)
        normal = torch.from_numpy(normal).view(1, 3, H, W)
        layout_seg = torch.from_numpy(layout_seg).view(1, 1, H, W)
        layout_depth = torch.from_numpy(layout_depth).view(1, 1, H, W)
        layout_normal = torch.from_numpy(layout_normal).view(1, 3, H, W)
        mask = torch.from_numpy(mask).view(1, 1, H, W)
        if cuda:
            extrinsic = extrinsic.cuda()
            intrinsic = intrinsic.cuda()
            image = image.cuda()
            seg = seg.cuda()
            depth = depth.cuda()
            normal = normal.cuda()
            layout_seg = layout_seg.cuda()
            layout_depth = layout_depth.cuda()
            layout_normal = layout_normal.cuda()
            mask = mask.cuda()
            
        rectified_extrinsic = utils.get_rectified_extrinsic(cuda, extrinsic)

        rectified_image, rectified_depth, rectified_normal, rectified_seg, rectified_mask = utils.reproject(cuda, \
            image, depth, normal, seg, mask, intrinsic, extrinsic, intrinsic[0], rectified_extrinsic[0])
        
        _, rectified_layout_depth, rectified_layout_normal, rectified_layout_seg, _ = utils.reproject(cuda, \
            image, layout_depth, layout_normal, layout_seg, mask, intrinsic, extrinsic, intrinsic[0], rectified_extrinsic[0])
        
        rate = rectified_mask.sum() / mask.sum()
        total_rate += rate
        print(rate)
        '''
        if rate < 0.4:
            continue
        '''
        
        rectified_extrinsic = rectified_extrinsic.view(4, 4).detach().cpu().numpy()
        rectified_image = rectified_image.view(3, H, W).detach().cpu().numpy()
        rectified_depth = rectified_depth.view(1, H, W).detach().cpu().numpy()
        rectified_normal = rectified_normal.view(3, H, W).detach().cpu().numpy()
        rectified_seg = rectified_seg.view(1, H, W).detach().cpu().numpy()
        rectified_layout_depth = rectified_layout_depth.view(1, H, W).detach().cpu().numpy()
        rectified_layout_normal = rectified_layout_normal.view(3, H, W).detach().cpu().numpy()
        rectified_layout_seg = rectified_layout_seg.view(1, H, W).detach().cpu().numpy()
        rectified_mask = rectified_mask.view(1, H, W).detach().cpu().numpy()
        save_one_group(base_dir, scene_id, id, base_name, rectified_extrinsic, rectified_image, rectified_seg, rectified_depth, rectified_normal, \
            rectified_layout_seg, rectified_layout_depth, rectified_layout_normal, rectified_mask)

    print(scene_id, 'finished!')


    total_rate = total_rate / len(id_list)
    print('Total rate', total_rate)


def main():
    """The main function of image rectification
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir', default='/home1/sgl/scannet_mine', type=str)
    parser.add_argument('--scene_id', default='scene0000_01', type=str)
    parser.add_argument('--cuda', default=1, type=int)
    args = parser.parse_args()
    rectify_one_scene(args.cuda, args.base_dir, args.scene_id)

if __name__ == "__main__":
    main()