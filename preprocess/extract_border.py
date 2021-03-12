import numpy as np
import cv2
import sys
import os
from scipy.optimize import leastsq
from plyfile import PlyData, PlyElement
import json
import glob
from math import *
from extract_point import *

def getPlaneIndex(lineIndex, planes):
    return lineIndex // planes, lineIndex % planes

def getLineIndex(planeIndexA, planeIndexB, planes):
    mini = min(planeIndexA, planeIndexB)
    maxi = max(planeIndexA, planeIndexB)
    return mini * planes + maxi

def getDist(a, b):
    return sqrt(np.sum((a - b) ** 2, axis = 0))

def getMinTree(points, threshold):
    '''
    description: get the min generated tree of the points
    parameter: points, threshold that the line will be splited
    return: the tree
    '''
    n = points.shape[0]
    edges = np.zeros((n, n))
    visit = [False] * n
    v_new = []
    dists = []
    e_new = []
    real_edge_lists = []
    for i in range(n):
        a = points[i]
        for j in range(n):
            b = points[j]
            edges[i][j] = getDist(a, b)

    v_new.append(0)
    visit[0] = True
    for ii in range(n - 1):
        min_dist = 1145141919810
        min_u = -1
        min_v = -1

        for i in [0, -1]:
            u = v_new[i]
            for v in range(n):
                if visit[v] == False:
                    dist = edges[u][v]
                    if dist < min_dist:
                        min_dist = dist
                        min_u = u
                        min_v = v
        visit[min_v] = True
        e_new.append((min_u, min_v))
        if min_u == v_new[0]:
            v_new.insert(0, min_v)
            dists.insert(0, min_dist)
        elif min_u == v_new[-1]:
            v_new.append(min_v)
            dists.append(min_dist)
    cutting_places = []
    for i in range(len(v_new) - 1):
        if dists[i] > threshold:
            cutting_places.append(i)
    cutting_places.append(len(v_new) - 1)
    start = 0
    for end in cutting_places:
        real_edge_lists.append(v_new[start:end])
        start = end + 1
    return real_edge_lists


def getMaxEdgeLength(points, faces):
    '''
    description: get the max edge length of all faces
    parameter: points, faces
    return: max_length
    '''
    max_length = 0
    for faceIndex in range(faces.shape[0]):
        face = faces[faceIndex]
        a = face[0]
        b = face[1]
        c = face[2]
        point_a = points[a]
        point_b = points[b]
        point_c = points[c]
        dist_ab = getDist(point_a, point_b)
        dist_ac = getDist(point_a, point_c)
        dist_bc = getDist(point_b, point_c)
        if dist_ab > max_length:
            max_length = dist_ab
        if dist_ac > max_length:
            max_length = dist_ac
        if dist_bc > max_length:
            max_length = dist_bc
    return max_length

def getBorder(planes, planeSegmentation, points, faces):
    '''
    description: After get the points of all segmented planes, get the border lines(both points and geometry approximation)
    parameter: plane_segmentation, points, faces
    return: border points, border line parameter
    '''
    max_edge_dist = getMaxEdgeLength(points, faces)
    lines = []
    useful_line_indexs = []
    borders = []
    for i in range(planes ** 2):
        lines.append(set())

    #get the points of all borders
    for faceIndex in range(faces.shape[0]):
        face = faces[faceIndex]
        a = face[0]
        b = face[1]
        c = face[2]
        if planeSegmentation[a] != planeSegmentation[b]:
            lineIndex = getLineIndex(planeSegmentation[a], planeSegmentation[b], planes)
            new_tuple = (min(a, b), max(a, b))
            lines[lineIndex].add(new_tuple)

        if planeSegmentation[a] != planeSegmentation[c]:
            lineIndex = getLineIndex(planeSegmentation[a], planeSegmentation[c], planes)
            new_tuple = (min(a, c), max(a, c))
            lines[lineIndex].add(new_tuple)

        if planeSegmentation[b] != planeSegmentation[c]:
            lineIndex = getLineIndex(planeSegmentation[b], planeSegmentation[c], planes)
            new_tuple = (min(b, c), max(b, c))
            lines[lineIndex].add(new_tuple)

    #get all center points
    for i in range(len(lines)):
        if len(lines[i]) > 0:
            plane_a, plane_b = getPlaneIndex(i, planes)
            useful_line_indexs.append(getPlaneIndex(i, planes))
            new_point_list = []
            for index_a, index_b in lines[i]:
                point_a = points[index_a]
                point_b = points[index_b]
                point_middle = (point_a + point_b) / 2
                new_point_list.append(point_middle)
            new_point_list = np.array(new_point_list)
            new_point_dict = {"plane_a" : plane_a, "plane_b" : plane_b, "points" : new_point_list}
            borders.append(new_point_dict)

    #use min_tree to get the lines
    for i in range(len(borders)):
        points = borders[i]["points"]
        min_trees = getMinTree(points, max_edge_dist)
        edge_list = []
        crucial_point_indexs_real = []
        for tree in min_trees:
            #get edges
            for j in range(len(tree) - 1):
                edge_list.append((tree[j], tree[j + 1]))

            #get crucial points
            min_tree_points = []
            length_tree = len(tree)
            for j in range(length_tree):
                min_tree_points.append(points[tree[j]])
            crucial_point_indexs = douglasPeucker(min_tree_points, 0, length_tree - 1, max_edge_dist / 10)
            for index in crucial_point_indexs:
                crucial_point_indexs_real.append(tree[index])

        borders[i]['edges'] = edge_list
        borders[i]['crucial_points'] = crucial_point_indexs_real
    
    all_new_points = []
    all_new_edges = []
    all_crucial_points = []
    for i in range(len(borders)):
        points = borders[i]['points']
        edges = borders[i]['edges']
        crucial_points = borders[i]['crucial_points']
        current_point_num_base = len(all_new_points)
        for point in points:
            all_new_points.append(point)
        for edge in edges:
            a, b = edge
            new_a = a + current_point_num_base
            new_b = b + current_point_num_base
            all_new_edges.append((new_a, new_b))
        for index in crucial_points:
            all_crucial_points.append(index + current_point_num_base)
    return all_new_points, all_new_edges, all_crucial_points

        

    
    