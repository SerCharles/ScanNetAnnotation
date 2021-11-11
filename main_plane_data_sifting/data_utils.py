"""The data util functions of main plane sifting
"""
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def load_rgb_image(file_name):
    """load a RGB image

    Args:
        file_name [str]: [the place of the RGB image]

    Returns:
        pic [PIL data]: [the PIL data of the image]
    """
    fp = open(file_name, 'rb')
    pic = Image.open(fp)
    pic_array = np.array(pic)
    fp.close()
    pic = Image.fromarray(pic_array)
    return pic

def load_grey_scale_image(file_name):
    """load a grey scale image(depth/norm/segmentation)

    Args:
        file_name [str]: [the place of the image]

    Returns:
        pic [PIL data]: [the PIL data of the image]
    """
    fp = open(file_name, 'rb')
    pic = Image.open(fp)
    pic_array = np.array(pic)
    fp.close()
    pic = Image.fromarray(pic_array)
    pic = pic.convert("I")
    return pic

def load_intrinsic(file_name):
    """load an intrinsic file of ScanNet

    Args:
        file_name [str]: [the place of the intrinsic file]

    Returns:
        intrinsic [numpy float array], [3 * 3]: [the intrinsic array]
    """
    intrinsic = np.zeros((3, 3), dtype = float)
    intrinsic[2][2] = 1.0

    f = open(file_name, 'r')
    lines = f.read().split('\n')
    for line in lines: 
        words = line.split()
        if len(words) > 0:
            if words[0] == 'm_calibrationColorIntrinsic':
                intrinsic[0][0] = float(words[2])
                intrinsic[1][1] = float(words[7])
                intrinsic[2][0] = float(words[4])
                intrinsic[2][1] = float(words[8])

    f.close()
    return intrinsic

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

def get_mask(depth, normal):
    """Get the mask of one picture, masking out those place with useless data
        H: the height of the picture
        W: the width of the picture
        
    Args:
        depth [torch float array], [1 * H * W]: [depth map]
        normal [torch float array], [3 * H * W]: [normal map]

    Returns:
        mask [torch boolean array], [1 * H * W]: [the mask of the data, 1 means ok, 0 means useless]
    """
    mask_depth = depth > 0 
    sqrt_normal = normal[0:1, :, :] ** 2 + normal[1:2, :, :] ** 2 + normal[2:3, :, :] ** 2
    mask_normal = ~(sqrt_normal < 1e-8)
    mask = mask_depth & mask_normal
    return mask

def resize_intrinsics(new_size, old_size, intrinsic):
    """Resize an intrinsic file

    Args:
        new_size [int array]: [height and width of the resized picture]
        old_size [int array]: [height and width of the original picture]
        intrinsic [torch float array], [3 * 3]: [the intrinsic array]

    Returns:
        intrinsic [torch float array], [3 * 3]: [the resized intrinsic array]
    """
    new_H = new_size[0]
    new_W = new_size[1]
    old_H = old_size[1]
    old_W = old_size[0]
    intrinsic[0][0] = intrinsic[0][0] / old_W * new_W
    intrinsic[2][0] = intrinsic[2][0] / old_W * new_W
    intrinsic[1][1]= intrinsic[1][1] / old_H * new_H
    intrinsic[2][1] = intrinsic[2][1] / old_H * new_H
    return intrinsic


def transform_vp_data(new_size, old_size, vanishing_point, whether_boundary):
    """Transform the vanishing point data
        H: the height of the picture
        W: the width of the picture
        
    Args:
        new_size [int array]: [height and width of the resized picture]
        old_size [int array]: [height and width of the original picture]
        vanishing_point [torch float array], [2]: [the vanishing points]
        whether_boundary [torch boolean array], [W]: [whether there are boundaries]

    Returns:
        vanishing_point [torch float array], [2]: [the transformed vanishing points]
        new_boundary [torch boolean array], [W]: [whether there are boundaries, transformed]
    """
    new_H = new_size[0]
    new_W = new_size[1]
    old_H = old_size[1]
    old_W = old_size[0]
    vanishing_point[0] = vanishing_point[0] / old_H * new_H
    vanishing_point[1] = vanishing_point[1] / old_W * new_W
    
    #interpolate the results
    index = torch.from_numpy(np.arange(0, old_W))
    i_boundary = torch.masked_select(index, whether_boundary)
    i_boundary = torch.clamp((i_boundary / old_W * new_W).long(), min=0, max=new_W - 1)
    new_boundary = torch.zeros(new_W, dtype=torch.bool)
    new_boundary[i_boundary] = True 

    return vanishing_point, new_boundary
        


def load_one_picture(base_dir, scene_id, id):
    """Load the group of data of one picture
        H: the height of the picture
        W: the width of the picture
        
    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the scene id to be handled]
        id [int]: [the id of the picture]
        
    Return:
        base_name [string]: [the base name of the group]
        intrinsic [torch float array], [3 * 3]: [the intrinsic of the group]
        extrinsic [torch float array], [4 * 4]: [the extrinsic of the group]
        image [torch float array], [3 * H * W]: [the image of the group]
        depth [torch float array], [1 * H * W]: [the depth of the group]
        normal [torch float array], [3 * H * W]: [the normal of the group]
        seg [torch boolean array], [1 * H * W]: [the segmentation of the group, 1 is background, 0 is empty]
        layout_seg [torch int array], [1 * H * W]: [the layout segmentation of the group]
        vanishing_point [torch float array], [2]: [the vanishing points of the group]
        whether_boundary [torch boolean array], [W]: [whether there are boundaries of the group]
        mask [torch boolean array], [1 * H * W]: [the mask of the group, masking which pixels are useful or not]
    """
    base_name = scene_id + '_' + str(id)
    intrinsic_name = '_info.txt'
    extrinsic_name = scene_id + "_" + str(id) + '.txt'
    image_name = scene_id + '_' + str(id) + '.jpg'
    depth_name = scene_id + '_' + str(id) + '.png'
    nx_name = scene_id + '_' + str(id) + '_nx.png'
    ny_name = scene_id + '_' + str(id) + '_ny.png'
    nz_name = scene_id + '_' + str(id) + '_nz.png'
    seg_name = scene_id + '_' + str(id) + '.png'
    layout_depth_name = scene_id + '_' + str(id) + '.png'
    layout_nx_name = scene_id + '_' + str(id) + '_nx.png'
    layout_ny_name = scene_id + '_' + str(id) + '_ny.png'
    layout_nz_name = scene_id + '_' + str(id) + '_nz.png'
    layout_seg_name = scene_id + '_' + str(id) + '.png'
    vp_name = scene_id + '_' + str(id) + '.npz'

    #get full name
    intrinsic_name = os.path.join(base_dir, scene_id, intrinsic_name)
    extrinsic_name = os.path.join(base_dir, scene_id, 'pose', extrinsic_name)
    image_name = os.path.join(base_dir, scene_id, 'color', image_name)
    depth_name = os.path.join(base_dir, scene_id, 'depth', depth_name)
    nx_name = os.path.join(base_dir, scene_id, 'norm', nx_name)
    ny_name = os.path.join(base_dir, scene_id, 'norm', ny_name)
    nz_name = os.path.join(base_dir, scene_id, 'norm', nz_name)
    seg_name = os.path.join(base_dir, scene_id, 'seg', seg_name)
    layout_depth_name = os.path.join(base_dir, scene_id, 'layout_depth', layout_depth_name)
    layout_nx_name = os.path.join(base_dir, scene_id, 'layout_norm', layout_nx_name)
    layout_ny_name = os.path.join(base_dir, scene_id, 'layout_norm', layout_ny_name)
    layout_nz_name = os.path.join(base_dir, scene_id, 'layout_norm', layout_nz_name)
    layout_seg_name = os.path.join(base_dir, scene_id, 'layout_seg', layout_seg_name) 
    vp_name = os.path.join(base_dir, scene_id, 'vanishing_point', vp_name)
    
    #load data
    intrinsic = torch.from_numpy(load_intrinsic(intrinsic_name)) #3 * 3
    extrinsic = torch.from_numpy(load_extrinsic(extrinsic_name)) #4 * 4
    image = load_rgb_image(image_name)
    depth = load_grey_scale_image(depth_name)
    nx = load_grey_scale_image(nx_name)
    ny = load_grey_scale_image(ny_name)
    nz = load_grey_scale_image(nz_name)
    seg = load_grey_scale_image(seg_name)
    layout_seg = load_grey_scale_image(layout_seg_name)
    vp_data = np.load(vp_name)
    vanishing_point = vp_data['vanishing_point']
    whether_boundary = vp_data['whether_boundaries']
    old_size = image.size
    new_size = [256, 256]
    vanishing_point = torch.from_numpy(vanishing_point)
    whether_boundary = torch.from_numpy(whether_boundary[0: old_size[0]])

    #transform data    
    transform_float = transforms.Compose([transforms.Resize(new_size, interpolation=Image.BILINEAR), transforms.ToTensor()])
    transform_int = transforms.Compose([transforms.Resize(new_size, interpolation=Image.NEAREST), transforms.ToTensor()])
    intrinsic = resize_intrinsics(new_size, old_size, intrinsic)
    image = transform_float(image)
    depth = transform_int(depth).float() / 1000.0
    nx = transform_int(nx).float() / 32768.0 - 1
    ny = transform_int(ny).float() / 32768.0 - 1
    nz = transform_int(nz).float() / 32768.0 - 1
    normal = torch.cat((nx, ny, nz), dim = 0)   
    seg = transform_int(seg)
    layout_seg = transform_int(layout_seg)
    vanishing_point, whether_boundary = transform_vp_data(new_size, old_size, vanishing_point, whether_boundary)
    mask = get_mask(depth, normal)
    seg = seg * mask
    return base_name, intrinsic, extrinsic, image, depth, normal, seg, layout_seg, vanishing_point, whether_boundary, mask
    

def add_boundary(image, vanishing_point, whether_boundary):
    """Add boundary information to the image to be printed

    Args:
        image [torch boolean array], [1 * H * W] or [torch float array], [3 * H * W]: [the original image]
        vanishing_point [torch float array], [2]: [the vanishing points of the group]
        whether_boundary [torch boolean array], [W]: [whether there are boundaries of the group]
        
    Returns
        image [torch boolean array], [1 * H * W] or [torch float array], [3 * H * W]: [the processed image]
    """
    C, H, W = image.size()
    index = torch.from_numpy(np.arange(0, W))
    boundary_places = torch.masked_select(index, whether_boundary)
    for i in boundary_places:
        dx = (vanishing_point[1] - i) / vanishing_point[0]
        current_x = float(i) 
        for j in range(H):
            place_x = int(current_x)
            if C == 1:
                image[0, j, place_x] = True 
            else: 
                image[0, j, place_x] = 0.99
            current_x += dx 
    return image
            
        

def save_one_picture(base_dir, scene_id, id, save_dir, image, vanishing_point, whether_boundary, wall_segs):
    """Save the information of one picture 
        N: the wall numbers
        H: the height of the picture
        W: the width of the picture
    Args:
        base_dir [string]: [the base directory of our modified ScanNet dataset]
        scene_id [string]: [the scene id to be handled]
        id [int]: [the id of the picture]
        save_dir [string]: [the save directory]
        image [torch float array], [3 * H * W]: [the image of the group]
        vanishing_point [torch float array], [2]: [the vanishing points of the group]
        whether_boundary [torch boolean array], [W]: [whether there are boundaries of the group]
        wall_segs [torch boolean array], [N * 1 * H * W]: [the segmentation information of one wall]
    """
    base_name = scene_id + '_' + str(id)
    save_dir_this = os.path.join(save_dir, base_name)
    if not os.path.exists(save_dir_this):
        os.mkdir(save_dir_this)
        
    N, _, H, W = wall_segs.size()
    total_wall_seg = torch.zeros((1, H, W), dtype=bool)
    for i in range(N):
        total_wall_seg = total_wall_seg | wall_segs[i]
        this_seg_name = base_name + '_' + str(i) + '_seg.png'
        this_seg_np = base_name + '_' + str(i) + '_seg.npy'
        this_seg_name_full = os.path.join(save_dir_this, this_seg_name)
        this_seg_np_full = os.path.join(save_dir_this, this_seg_np)
        np.save(this_seg_np_full, wall_segs[i].numpy())
        this_seg = add_boundary(wall_segs[i], vanishing_point, whether_boundary)
        this_seg = (this_seg.view(H, W) * 60000).numpy().astype(np.uint16)
        picture_seg = Image.fromarray(this_seg)
        picture_seg.save(this_seg_name_full)

    total_seg_name = base_name + '_total_seg.png'
    total_seg_name_full = os.path.join(save_dir_this, total_seg_name)
    total_wall_seg = add_boundary(total_wall_seg, vanishing_point, whether_boundary)
    total_seg = (total_wall_seg.view(H, W) * 60000).numpy().astype(np.uint16)
    picture_total_seg = Image.fromarray(total_seg)
    picture_total_seg.save(total_seg_name_full)
    
    if not os.path.exists(os.path.join(save_dir, 'pictures')):
        os.mkdir(os.path.join(save_dir, 'pictures'))
    picture_name = base_name + '.jpg'
    picture_name_full = os.path.join(save_dir_this, picture_name)
    picture_name_original = os.path.join(save_dir, 'pictures', scene_id + '_' + str(id) + '.jpg')
    image_save = torch.clamp(image.permute(1, 2, 0) * 256, 0, 255).numpy().astype(np.uint8)
    picture_original = Image.fromarray(image_save)
    picture_original.save(picture_name_original)

    picture = add_boundary(image, vanishing_point, whether_boundary)
    picture = torch.clamp(picture.permute(1, 2, 0) * 256, 0, 255).numpy().astype(np.uint8)
    picture = Image.fromarray(picture)
    picture.save(picture_name_full)
    






