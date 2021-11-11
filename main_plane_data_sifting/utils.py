"""Functions used in main plane sifting
"""

import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from math import *
import torch

#pixels and planes
def normalize(norm, epsilon=1e-8):
    """Function used in normalizing a vector
        H: the height of the picture
        W: the width of the picture

    Args:
        norm [torch float array], [3 * H * W]: [the vector to be normalized]
        epsilon [float], [1e-8 by default]: [used in preventing /0]

    Returns:
        norm [torch float array], [3 * H * W]: [normalized vector]
    """
    nx = norm[0:1, :, :] #1 * H * W
    ny = norm[1:2, :, :] #1 * H * W
    nz = norm[2:3, :, :] #1 * H * W
    length = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2) #1 * H * W
    norm = norm / (length + epsilon)
    return norm


def get_plane_info_per_pixel(norm, depth, intrinsic, epsilon=1e-8):
    """Given the normal and depth information per pixel, get the plane formula of each pixel based on it
        H: the height of the picture
        W: the width of the picture

    Args:
        norm [torch float array], [3 * H * W]: [the predicted normal per pixel]
        depth [torch float array], [1 * H * W]: [the predicted depth per pixel]
        intrinsic [torch float array], [3 * 3]: [the intrinsic of picture]
        epsilon [float], [1e-8 by default]: [a small number to avoid / 0]

    Returns:
        plane_info_per_pixel [torch float array], [4 * H * W]: [the plane info per pixel, (A, B, C, D) which satisfies Ax + By + Cz + D = 0]
    """
    _, H, W  = norm.size()
    xx, yy = np.meshgrid(np.array([ii for ii in range(W)]), np.array([ii for ii in range(H)]))
    xx = torch.from_numpy(xx)
    yy = torch.from_numpy(yy)
    
    xx = xx.view(1, H, W)#1 * H * W
    yy = yy.view(1, H, W)#1 * H * W
    fx = intrinsic[0, 0].view(1, 1, 1).repeat(1, H, W) #1 * H * W
    fy = intrinsic[1, 1].view(1, 1, 1).repeat(1, H, W) #1 * H * W
    x0 = intrinsic[2, 0].view(1, 1, 1).repeat(1, H, W) #1 * H * W
    y0 = intrinsic[2, 1].view(1, 1, 1).repeat(1, H, W) #1 * H * W
    x = ((xx - x0) / fx) * depth #1 * H * W
    y = ((yy - y0) / fy) * depth #1 * H * W

    norm = normalize(norm, epsilon=epsilon)
    A = norm[0:1, :, :] #1 * H * W
    B = norm[1:2, :, :] #1 * H * W
    C = norm[2:3, :, :] #1 * H * W
    D = - A * x - B * y - C * depth #1 * H * W
    plane_info_per_pixel = torch.cat((A, B, C, D), dim = 0) #4 * H * W
    return plane_info_per_pixel

def get_norm_and_depth_per_pixel(plane_info_per_pixel, intrinsic, epsilon=1e-8):
    """Given the normal and depth information per pixel, get the plane formula of each pixel based on it
        H: the height of the picture
        W: the width of the picture

    Args:
        plane_info_per_pixel [torch float array], [4 * H * W]: [the plane info per pixel, (A, B, C, D) which satisfies Ax + By + Cz + D = 0]
        intrinsic [torch float array], [3 * 3]: [the intrinsic of picture]
        epsilon [float], [1e-8 by default]: [a small number to avoid / 0]
    Returns:
        norm [torch float array], [3 * H * W]: [the normal per pixel]
        depth [torch float array], [1 * H * W]: [the depth per pixel]
    """
    _, H, W  = plane_info_per_pixel.size()
    A = plane_info_per_pixel[0:1, :, :] #1 * H * W
    B = plane_info_per_pixel[1:2, :, :] #1 * H * W
    C = plane_info_per_pixel[2:3, :, :] #1 * H * W
    D = plane_info_per_pixel[3:4, :, :] #1 * H * W
    norm = torch.cat((A, B, C), dim=1)
    norm = normalize(norm, epsilon)

    xx, yy = np.meshgrid(np.array([ii for ii in range(W)]), np.array([ii for ii in range(H)]))
    xx = torch.from_numpy(xx)
    yy = torch.from_numpy(yy)

    
    xx = xx.view(1, H, W)#1 * H * W
    yy = yy.view(1, H, W)#1 * H * W
    fx = intrinsic[0, 0].view(1, 1, 1).repeat(1, H, W) #1 * H * W
    fy = intrinsic[1, 1].view(1, 1, 1).repeat(1, H, W) #1 * H * W
    x0 = intrinsic[2, 0].view(1, 1, 1).repeat(1, H, W) #1 * H * W
    y0 = intrinsic[2, 1].view(1, 1, 1).repeat(1, H, W) #1 * H * W
    x_z = ((xx - x0) / fx) #1 * H * W   #X / Z
    y_z = ((yy - y0) / fy) #1 * H * W   #Y / Z

    divide = -(A * x_z + B * y_z + C) #1 * H * W   #Y / Z
    divide = divide + epsilon * torch.eq(divide, 0) #trick, avoid / 0
    depth = D / divide
    return norm, depth


def get_average_plane_info_from_pixels(plane_info_per_pixel, plane_seg):
    """Given the average plane info of the planes of pixels and the segmentation mask, get the average plane info of all planes
        H: the height of the picture
        W: the width of the picture
        M: the biggest possible segmentation id

    Args:
        plane_info_per_pixel [torch float array], [4 * H * W]: [the plane info per pixel, (A, B, C, D) which satisfies Ax + By + Cz + D = 0]
        plane_seg [torch int array], [1 * H * W]: [the segmentation label per pixel]

    Returns:
        average_plane_infos [torch float array], [(M + 1) * 4]: [the average plane info of the plane, (A, B, C, D) which satisfies Ax + By + Cz + D = 0]
    """
    M = int(torch.max(plane_seg))
    _, H, W  = plane_info_per_pixel.size()

    average_plane_infos = []
    for i in range(0, M + 1):
        mask_batch = torch.eq(plane_seg, i).repeat(4, 1, 1) #1 * H * W
        the_total = torch.sum(plane_info_per_pixel * mask_batch, dim=[1, 2]) #4
        useful_sum = torch.sum(mask_batch) #1
        new_count = useful_sum + torch.eq(useful_sum, 0) #trick to avoid / 0
        new_average_plane_info = (the_total / new_count).unsqueeze(0) #4
        average_plane_infos.append(new_average_plane_info)
    average_plane_infos = torch.cat(average_plane_infos, dim=0) #(M + 1) * 4
    return average_plane_infos

def set_average_plane_info_to_pixels(plane_seg, average_plane_info):
    """Set the plane info of all the pixels to be the average info of their planes
        H: the height of the picture
        W: the width of the picture
        M: the biggest possible segmentation id

    Args:
        plane_seg [torch int array], [1 * H * W]: [the segmentation label per pixel]
        average_plane_info [torch float array], [(M + 1) * 4]: [the average plane info of the plane, (A, B, C, D) which satisfies Ax + By + Cz + D = 0]

    Returns:
        plane_info_per_pixel [torch float array], [4 * H * W]: [the plane info per pixel, (A, B, C, D) which satisfies Ax + By + Cz + D = 0]
    """
    _, H, W = plane_seg.size()
    M =  average_plane_info.size(0) - 1

    plane_info_per_pixel = []
    for the_id in range(M + 1):
        mask = torch.eq(plane_seg[0], the_id) #H * W
        a = average_plane_info[the_id][0] #1
        b = average_plane_info[the_id][1] #1
        c = average_plane_info[the_id][2] #1
        d = average_plane_info[the_id][3] #1
        masked_a = (mask * a).unsqueeze(0) #1 * H * W
        masked_b = (mask * b).unsqueeze(0) #1 * H * W
        masked_c = (mask * c).unsqueeze(0) #1 * H * W
        masked_d = (mask * d).unsqueeze(0) #1 * H * W
        the_plane_info = torch.cat([masked_a, masked_b, masked_c, masked_d]).unsqueeze(0) #1 * 4 * H * W
        plane_info_per_pixel.append(the_plane_info)

    plane_info_per_pixel = torch.cat(plane_info_per_pixel) #(M + 1) * 4 * H * W, the result plane info of each planes of batch i
    plane_info_per_pixel = torch.sum(plane_info_per_pixel, dim=0, keepdim=False) #4 * H * W, the result plane info of batch i
    return plane_info_per_pixel




#clustering
def mean_shift_clustering(plane_info):
    """The mean-shift clustering function
        N: the number of pixels/points whose plane formula are valid

    Args:
        plane_info [panda float array], [N * 4]: [the plane info of valid pixels/points]

    Returns:
        labels [numpy in array], [N]: [the clustering labels of valid pixels]
    """
    bandwidth = estimate_bandwidth(plane_info, quantile=0.1, n_samples=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(plane_info)
    labels = ms.labels_
    return labels

def get_useful_labels(labels, threshold):
    """Select the clustering labels whose sum are big enough
        N: the number of pixels whose plane formula are valid

    Args:
        labels [numpy int array], [N]: [the clustering labels of valid pixels]
        threshold [int]: [the threshold of pixels required for a segmentation label to be valid]

    Returns:
        labels [numpy int array], [N]: [the clustering labels of valid pixels after selection]
    """
    unique_labels = np.unique(labels)
    useful_labels = []
    for label in unique_labels:
        label_mask = (labels == label)
        label_num = (labels == label).sum()
        if label_num >= threshold:
            useful_labels.append(label)
        else: 
            labels[label_mask] = 0
    return labels

def clustering(mask, plane_info, threshold_ratio=0.05):
    """Clustering the pixels by their plane infos 
        H: the height of the picture
        W: the width of the picture

    Args:
        mask [torch boolean array], [1 * H * W]: [whether the pixel/point's plane info is calculated or not, if yes, process it]
        plane_info [torch float array], [4 * H * W]: [the plane infos]
        threshold_ratio [float], [0.05 by default]: [the threshold of the ratio of pixels/points required for a segmentation label to be valid]

    Returns:
        labels [torch int array], [1 * H * W]: [the clustering result of each pixels/points, 0 means no label]
    """
    _, H, W = mask.size()
    M = H * W
    mask = mask.view(1, M) #1 * M
    plane_info = plane_info.view(4, M) #1 * M

    #select only the pixels whose plane info are valid
    index_list = torch.linspace(0, M - 1, steps=M, dtype=np.int).view(1, M)

    selected_plane_info_a = torch.masked_select(plane_info[0:1, :], mask.bool()).unsqueeze(0) #1 * K, K is the number of valid points
    selected_plane_info_b = torch.masked_select(plane_info[1:2, :], mask.bool()).unsqueeze(0) #1 * K
    selected_plane_info_c = torch.masked_select(plane_info[2:3, :], mask.bool()).unsqueeze(0) #1 * K
    selected_plane_info_d = torch.masked_select(plane_info[3:4, :], mask.bool()).unsqueeze(0) #1 * K
    selected_plane_info = torch.cat((selected_plane_info_a, selected_plane_info_b, selected_plane_info_c, selected_plane_info_d), dim = 0) #4 * K
    selected_index = torch.masked_select(index_list, mask.bool()) #1 * K

    #switch torch to float and pandas
    selected_plane_info = selected_plane_info.numpy()
    selected_index = selected_index.numpy()
    plane_info_pandas = pd.DataFrame(selected_plane_info.T, columns=list('abcd'))

    #clustering
    selected_labels = mean_shift_clustering(plane_info_pandas) + 1

    #remove the labels with little pixels
    threshold = int(threshold_ratio * M)
    cleared_labels = get_useful_labels(selected_labels, threshold)

    #use the clustering result to get the segmentation info of all the pixels
    labels = np.zeros(M, dtype=int)
    labels[selected_index] = cleared_labels
    labels = torch.from_numpy(labels).view(1, H, W)
    return labels


#bounding box and measures
