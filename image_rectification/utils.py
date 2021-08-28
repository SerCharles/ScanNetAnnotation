"""Functions used in image rectification
"""

import numpy as np
from math import *
import torch



def rotate_pointcloud(cuda, pointcloud, pointcloud_norm, source_extrinsic=None, target_extrinsic=None):
    """Rotate the pointcloud and its normal
        N: the batch size
        NUM: the number of pointcloud
        M: the biggest possible segmentation id

    Args:
        cuda [boolean]: [Whether use gpu or not]
        pointcloud [torch float array], [N * 3 * NUM]: [the pointcloud in the source camera coordinate]
        pointcloud_norm [torch float array], [N * 3 * NUM]: [the pointcloud normal in the source camera coordinate]
        source_extrinsic [torch float array], [N * 4 * 4]: [the extrinsic(camera to world matrix) of source picture]. Defaults to I.
        target_extrinsic [torch float array], [N * 4 * 4]: [the extrinsic(camera to world matrix) of target picture]. Defaults to I.

    Returns:
        target_coordinate [torch float array], [N * 3 * NUM]: [the pointcloud in the target camera coordinate]
        target_norm [torch float array], [N * 3 * NUM]: [the pointcloud norm in the target camera coordinate]
    """
    N, _, NUM  = pointcloud.size()
    if source_extrinsic == None:
        source_extrinsic = torch.eye(4).view(1, 4, 4).repeat(N, 1, 1)
        if cuda:
            source_extrinsic = source_extrinsic.cuda()

    if target_extrinsic == None:
        target_extrinsic = torch.eye(4).view(1, 4, 4).repeat(N, 1, 1)
        if cuda:
            target_extrinsic = target_extrinsic.cuda()   

    #convert the source coordinate to the world coordinate
    rotation = source_extrinsic[:, :3, :3] #N * 3 * 3
    translation = source_extrinsic[:, :3, 3].view(N, 3, 1).repeat(1, 1, NUM) #N * 3 * NUM
    world_coordinate = torch.matmul(rotation, pointcloud) #N * 3 * NUM
    world_coordinate = world_coordinate + translation #N * 3 * NUM
    world_norm = torch.matmul(rotation, pointcloud_norm) #N * 3 * NUM

    #convert the world coordinate to the target coordinate
    rotation = torch.inverse(target_extrinsic[:, :3, :3]) #N * 3 * 3
    translation = -torch.matmul(rotation, target_extrinsic[:, :3, 3].view(N, 3, 1).repeat(1, 1, NUM)) #N * 3 * NUM
    target_coordinate = torch.matmul(rotation, world_coordinate) #N * 3 * NUM
    target_coordinate = target_coordinate + translation #N * 3 * NUM
    target_norm = torch.matmul(rotation, world_norm) #N * 3 * NUM
    return target_coordinate, target_norm


#pixel and pointcloud coversion
def pixel_to_pointcloud(cuda, source_depth, source_norm, source_intrinsic, source_extrinsic):
    """Convert the source picture to the pointcloud in the the world coordination, as well as the normal
        N: the batch size(target pictures to be handled at the same time)
        H: the height of the picture
        W: the width of the picture

    Args:
        cuda [boolean]: [use GPU or not]
        source_depth [torch float array], [N * 1 * H * W]: [the depth map of source picture] 
        source_norm [torch float array], [N * 3 * H * W]: [the norm map of source picture] 
        source_intrinsic [torch float array], [N * 3 * 3]: [the intrinsic of source picture] 
        source_extrinsic [torch float array], [N * 4 * 4]: [the extrinsic of source picture] 

    Returns:
        world_coordinate [torch float array], [N * 3 * (H * W)]: [the pointcloud]
        world_norm [torch float array], [N * 3 * (H * W)]: [the norm map of the target pointcloud]
    """
    #convert the source picture pixels to the source picture's camera coordinate
    N, C, H, W  = source_norm.size()
    xx, yy = np.meshgrid(np.array([ii for ii in range(W)]), np.array([ii for ii in range(H)]))
    xx = torch.from_numpy(xx)
    yy = torch.from_numpy(yy)
    if cuda:
        xx = xx.cuda()
        yy = yy.cuda()
    xx = xx.view(1, 1, H, W).repeat(N, 1, 1, 1) #N * 1 * H * W
    yy = yy.view(1, 1, H, W).repeat(N, 1, 1, 1) #N * 1 * H * W
    fx = source_intrinsic[:, 0, 0].view(N, 1, 1, 1).repeat(1, 1, H, W) #N * 1 * H * W
    fy = source_intrinsic[:, 1, 1].view(N, 1, 1, 1).repeat(1, 1, H, W) #N * 1 * H * W
    x0 = source_intrinsic[:, 2, 0].view(N, 1, 1, 1).repeat(1, 1, H, W) #N * 1 * H * W
    y0 = source_intrinsic[:, 2, 1].view(N, 1, 1, 1).repeat(1, 1, H, W) #N * 1 * H * W
    x = ((xx - x0) / fx) * source_depth 
    y = ((yy - y0) / fy) * source_depth 
    z = source_depth
    source_coordinate = torch.cat((x, y, z), dim=1).float() #N * 3 * H * W

    #convert the source picture's camera coordinate to the world coordinate
    rotation = source_extrinsic[:, :3, :3] #N * 3 * 3
    translation = source_extrinsic[:, :3, 3].view(N, 3, 1, 1).repeat(1, 1, H, W) #N * 3 * H * W
    world_coordinate = torch.matmul(rotation, source_coordinate.view(N, 3, H * W)).view(N, 3, H, W)
    world_coordinate = world_coordinate + translation #N * 3 * H * W
    world_norm = torch.matmul(rotation, source_norm.view(N, 3, H * W)).view(N, 3, H, W)
    world_coordinate = world_coordinate.view(N, 3, H * W)
    world_norm = world_norm.view(N, 3, H * W)
    return world_coordinate, world_norm


def pointcloud_to_index(coordinate, intrinsic, mask):
    """Switch the point cloud to x-y index in the target picture
        N: the batch size(target pictures to be handled at the same time)
        H: the height of the picture
        W: the width of the picture

    Args:
        coordinate [torch float array], [N * 3 * (H * W)]: [the point cloud in the target picture] 
        intrinsic [torch float array], [3 * 3]: [the intrinsic of target picture] 
        mask [torch boolean array], [N * 1 * H * W]: [the mask of source picture] 

    Returns:
        index_x [torch int array], [N * 1 * H * W]: [the x index in the target picture]
        index_y [torch int array], [N * 1 * H * W]: [the y index in the target picture]
        z [torch int array], [N * 1 * H * W]: [the depth in the target picture]
    """
    N, _, H, W = mask.size()
    coordinate = coordinate.view(N, 3, H, W)
    x = coordinate[:, 0:1, :, :] #N * 1 * H * W
    y = coordinate[:, 1:2, :, :] #N * 1 * H * W
    z = coordinate[:, 2:3, :, :] #N * 1 * H * W

    #convert the target picture's camera coordinate to the target picture's picture coordinate
    fx = intrinsic[0, 0].view(1, 1, 1, 1).repeat(N, 1, H, W) #N * 1 * H * W
    fy = intrinsic[1, 1].view(1, 1, 1, 1).repeat(N, 1, H, W) #N * 1 * H * W
    x0 = intrinsic[2, 0].view(1, 1, 1, 1).repeat(N, 1, H, W) #N * 1 * H * W
    y0 = intrinsic[2, 1].view(1, 1, 1, 1).repeat(N, 1, H, W) #N * 1 * H * W
    mask_z = (z > 0) & mask 
    z_new = (z * mask_z) + (~mask_z) #avoid /0
    index_x = ((x / z_new) * fx + x0).int() 
    index_y = ((y / z_new) * fy + y0).int()
    return index_x, index_y, z

def scatter_add_multiple_batch(cuda, source_features, index_total):
    """The re-implementation of torch.scatter_add, when the source features have multiple channels
        N: the batch size(target pictures to be handled at the same time)
        C: the channels of source features
        H: the height of the picture
        W: the width of the picture

    Args:
        cuda [boolean]: [Whether use gpu or not]
        source_features [torch float array], [(N * H * W) * C]: [the feature map of source picture, the src of torch.scatter_add] 
        index_total [torch long(int64) array], [(N * H * W) * 1]: [the index of source picture, the index of torch.scatter_add] 

    Returns:
        target_features [torch float array], [(N * H * W) * C]: [the result feature map of target picture]
    """
    target_features = []

    for i in range(source_features.size(1)):
        zeros = torch.zeros((source_features.size(0), 1), dtype=torch.float32)
        if cuda:
            zeros = zeros.cuda()
        new_item = torch.scatter_add(zeros, 0, index_total, source_features[:, i:i + 1].float())
        target_features.append(new_item)
    target_features = torch.stack(target_features, dim=1)
    return target_features

def scatter_multiple_batch(cuda, source_features, index_total):
    """The re-implementation of torch.scatter, when the source features have multiple channels
        N: the batch size(target pictures to be handled at the same time)
        C: the channels of source features
        H: the height of the picture
        W: the width of the picture

    Args:
        cuda [boolean]: [Whether use gpu or not]
        source_features [torch float array], [(N * H * W) * C]: [the feature map of source picture, the src of torch.scatter] 
        index_total [torch long(int64) array], [(N * H * W) * 1]: [the index of source picture, the index of torch.scatter] 

    Returns:
        target_features [torch float array], [(N * H * W) * C]: [the result feature map of target picture]
    """
    target_features = []

    for i in range(source_features.size(1)):
        zeros = torch.zeros((source_features.size(0), 1), dtype=torch.float32)
        if cuda:
            zeros = zeros.cuda()
        new_item = torch.scatter(zeros, 0, index_total, source_features[:, i:i + 1].float())
        target_features.append(new_item)
    target_features = torch.stack(target_features, dim=1)
    return target_features


def reproject(cuda, source_feature, source_depth, source_norm, source_seg, source_mask, source_intrinsic, source_extrinsic, \
    target_intrinsic, target_extrinsic):
    """Reproject the feature, depth and norm of one picture to another
        N: the batch size(target pictures to be handled at the same time)
        H: the height of the picture
        W: the width of the picture

    Args:
        args [arguments]: [the global arguments]
        source_feature [torch float array], [N * C * H * W]: [the feature map of source picture] 
        source_depth [torch float array], [N * 1 * H * W]: [the depth map of source picture] 
        source_norm [torch float array], [N * 3 * H * W]: [the norm map of source picture] 
        source_seg [torch int array], [N * 1 * H * W]: [the seg map of source picture] 
        source_mask [torch boolean array], [N * 1 * H * W]: [the mask of source picture] 
        source_intrinsic [torch float array], [N * 3 * 3]: [the intrinsic of source picture] 
        source_extrinsic [torch float array], [N * 4 * 4]: [the extrinsic of source picture] 
        target_intrinsic [torch float array], [3 * 3]: [the intrinsic of target picture]
        target_extrinsic [torch float array], [4 * 4]: [the extrinsic of target picture]

    Returns:
        target_feature [torch float array], [N * C * H * W]: [the feature map of target picture]
        target_depth [torch float array], [N * 1 * H * W]: [the depth map of target picture]
        target_norm [torch float array], [N * 3 * H * W]: [the norm map of target picture]
        target_seg [torch boolean array], [N * 1 * H * W]: [the seg of target picture]
        target_mask [torch boolean array], [N * 1 * H * W]: [the mask of target picture]
    """
    N, C, H, W  = source_feature.size()
    world_coordinate, world_norm = pixel_to_pointcloud(cuda, source_depth, source_norm, source_intrinsic, source_extrinsic)
    target_coordinate, target_norm = rotate_pointcloud(cuda, world_coordinate, world_norm, target_extrinsic=target_extrinsic.view(1, 4, 4).repeat(N, 1, 1))
    target_norm = target_norm.view(N, 3, H, W)
    index_x, index_y, z = pointcloud_to_index(target_coordinate, target_intrinsic, source_mask)
    index_y = index_y.permute(0, 2, 3, 1).view(N * H * W, 1).long()
    index_x = index_x.permute(0, 2, 3, 1).view(N * H * W, 1).long()

    index_batch = torch.linspace(0, N - 1, steps=N).int().view(N, 1, 1, 1).repeat(1, H, W, 1).view(N * H * W, 1).long()
    ones = torch.ones((N * H * W, 1), dtype=torch.int32)
    if cuda:
        index_batch = index_batch.cuda()
        ones = ones.cuda()

    useful_mask = (index_x >= 0) & (index_x < W) & (index_y >= 0) & (index_y < H) #(N * H * W) * 1
    index_x = (index_x * useful_mask).long()
    index_y = (index_y * useful_mask).long()
    index_total = index_batch * (H * W) + (index_y * W) + index_x #(N * H * W) * 1
    index_total = index_total * useful_mask
    
    source_modified_feature = torch.reshape(source_feature.permute(0, 2, 3, 1), (N * H * W, C)) * useful_mask
    source_modified_depth = torch.reshape(z.permute(0, 2, 3, 1), (N * H * W, 1)) * useful_mask
    source_modified_normal = torch.reshape(target_norm.permute(0, 2, 3, 1), (N * H * W, 3)) * useful_mask
    source_modified_seg = torch.reshape(source_seg.permute(0, 2, 3, 1), (N * H * W, 1)) * useful_mask
    source_modified_mask = ones * useful_mask #(N * H * W) * 1

    target_picture_feature = scatter_add_multiple_batch(cuda, source_modified_feature, index_total)
    target_picture_depth = scatter_add_multiple_batch(cuda, source_modified_depth, index_total)
    target_picture_normal = scatter_add_multiple_batch(cuda, source_modified_normal, index_total)
    target_picture_seg = scatter_multiple_batch(cuda, source_modified_seg, index_total)
    target_picture_num = scatter_add_multiple_batch(cuda, source_modified_mask, index_total)

    target_picture_mask = torch.gt(target_picture_num, 0)
    target_picture_num = target_picture_num + (~target_picture_mask)
    target_picture_feature = target_picture_feature / target_picture_num
    target_picture_normal = target_picture_normal / target_picture_num
    target_picture_depth = target_picture_depth / target_picture_num
    
    target_picture_feature = target_picture_feature * target_picture_mask
    target_picture_normal = target_picture_normal * target_picture_mask
    target_picture_depth = target_picture_depth * target_picture_mask
    target_picture_seg = target_picture_seg * target_picture_mask

    target_picture_feature = target_picture_feature.view(N, H, W, C).permute(0, 3, 1, 2)
    target_picture_normal = target_picture_normal.view(N, H, W, 3).permute(0, 3, 1, 2)
    target_picture_depth = target_picture_depth.view(N, H, W, 1).permute(0, 3, 1, 2)
    target_picture_seg = target_picture_seg.view(N, H, W, 1).permute(0, 3, 1, 2)
    target_picture_mask = target_picture_mask.view(N, H, W, 1).permute(0, 3, 1, 2)
    return target_picture_feature, target_picture_depth, target_picture_normal, target_picture_seg, target_picture_mask


def get_rectified_extrinsic(cuda, original_extrinsic):
    """Get the rectified extrinsic of pictures, used in picture rectification
        N: the batch size

    Args:
        cuda [boolean]: whether use gpu or not
        original_extrinsic [torch float array], [N * 4 * 4]: [the original extrinsic of the pictures]

    Returns:
        rectified_extrinsic [torch float array], [N * 4 * 4]: [the rectified extrinsic of the pictures]
    """
    N = original_extrinsic.size(0)
    rectified_extrinsic = torch.zeros((N, 4, 4), dtype=torch.float32)
    if cuda:
        rectified_extrinsic = rectified_extrinsic.cuda()
    for i in range(N):
        extrinsic = original_extrinsic[i]
        rotation = extrinsic[:3, :3] #3 * 3
        translation = extrinsic[:3, 3] #3

        #set translation
        rectified_extrinsic[i, 0, 3] = translation[0]
        rectified_extrinsic[i, 1, 3] = translation[1]
        rectified_extrinsic[i, 2, 3] = translation[2]

        #set constants
        rectified_extrinsic[i, 3, 3] = 1.0
        rectified_extrinsic[i, 2, 1] = -1.0

        #calculate approximate transformation
        r00 = rotation[0, 0] 
        r02 = rotation[0, 2]
        r10 = rotation[1, 0]
        r12 = rotation[1, 2]
        r20 = rotation[2, 0]
        r22 = rotation[2, 2]
        left_sum = torch.pow(r00, 2) + torch.pow(r10, 2)
        right_sum = torch.pow(r02, 2) + torch.pow(r12, 2)
        if left_sum > right_sum:
            new_r00 = r00 / torch.sqrt(torch.pow(r00, 2) + torch.pow(r10, 2))
            new_r10 = r10 / torch.sqrt(torch.pow(r00, 2) + torch.pow(r10, 2))
            new_r02 = torch.sqrt(1 - new_r00 ** 2)
            new_r12 = torch.sqrt(1 - new_r10 ** 2)
            if r02 < 0:
                new_r02 = -new_r02
            if r12 < 0:
                new_r12 = -new_r12
        else:
            new_r02 = r02 / torch.sqrt(torch.pow(r02, 2) + torch.pow(r12, 2))
            new_r12 = r12 / torch.sqrt(torch.pow(r02, 2) + torch.pow(r12, 2))
            new_r00 = torch.sqrt(1 - new_r02 ** 2)
            new_r10 = torch.sqrt(1 - new_r12 ** 2)
            if r00 < 0:
                new_r00 = -new_r00
            if r10 < 0:
                new_r10 = -new_r10
        rectified_extrinsic[i, 0, 0] = new_r00
        rectified_extrinsic[i, 0, 2] = new_r02
        rectified_extrinsic[i, 1, 0] = new_r10
        rectified_extrinsic[i, 1, 2] = new_r12

    return rectified_extrinsic