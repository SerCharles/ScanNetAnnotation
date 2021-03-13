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
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return sqrt(dx ** 2 + dy ** 2 + dz ** 2)

def getMinTree(points, threshold):
    '''
    description: get the min generated tree of the points
    parameter: points, threshold that the line will be splited
    return: the tree
    '''
    n = len(points)
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
    lines_bordered = []
    meshes_bordered = []
    useful_line_indexs = []
    for i in range(planes ** 2):
        lines_bordered.append(set())

    #get the points of all borders
    for faceIndex in range(faces.shape[0]):
        face = faces[faceIndex]
        a = face[0]
        b = face[1]
        c = face[2]

        #first consider that the mesh is divided by three different kinds
        type_a = planeSegmentation[a]
        type_b = planeSegmentation[b]
        type_c = planeSegmentation[c]

        
        if type_a != type_b and type_a != type_c and type_b != type_c:
            meshes_bordered.append((a, b, c))
            continue

        
        if type_a != type_b:
            lineIndex = getLineIndex(type_a, type_b, planes)
            new_tuple = (min(a, b), max(a, b))
            lines_bordered[lineIndex].add(new_tuple)

        if type_a != type_c:
            lineIndex = getLineIndex(type_a, type_c, planes)
            new_tuple = (min(a, c), max(a, c))
            lines_bordered[lineIndex].add(new_tuple)

        if type_b != type_c:
            lineIndex = getLineIndex(type_b, type_c, planes)
            new_tuple = (min(b, c), max(b, c))
            lines_bordered[lineIndex].add(new_tuple)


    all_new_points = []
    new_point_border_num = []
    all_new_edges = []
    all_crucial_points = []
    #get all points
    for i in range(len(lines_bordered)):
        if len(lines_bordered[i]) > 0:
            plane_a, plane_b = getPlaneIndex(i, planes)
            useful_line_indexs.append(getPlaneIndex(i, planes))
            new_point_list = []
            for index_a, index_b in lines_bordered[i]:
                point_a = points[index_a]
                point_b = points[index_b]
                point_middle = (point_a + point_b) / 2
                all_new_points.append(point_middle)
                new_point_border_num.append([i])

    for a, b, c in meshes_bordered:
        point_a = points[a]
        point_b = points[b]
        point_c = points[c]
        type_a = planeSegmentation[a]
        type_b = planeSegmentation[b]
        type_c = planeSegmentation[c]
        point_middle = (point_a + point_b + point_c) / 3
        all_new_points.append(point_middle)
        new_point_border_num.append([getLineIndex(type_a, type_b, planes), \
            getLineIndex(type_a, type_c, planes), getLineIndex(type_b, type_c, planes)])
    
    border_index = [-1] * (planes ** 2)
    borders = []
    for i in range(len(all_new_points)):
        for j in new_point_border_num[i]:
            if border_index[j] < 0:
                border_index[j] = len(borders)
                plane_a, plane_b = getPlaneIndex(j, planes)
                new_dict = {'plane_a': plane_a, 'plane_b': plane_b, 'points': []}
                borders.append(new_dict)
            place = border_index[j]
            borders[place]['points'].append(i)

    #use min_tree to get the lines
    for i in range(len(borders)):
        point_indexs = borders[i]["points"]
        current_points = []
        for index in point_indexs:
            current_points.append(all_new_points[index])
        min_trees = getMinTree(current_points, max_edge_dist)
        edge_list = []
        crucial_point_indexs_real = []
        for tree in min_trees:
            #get edges
            if len(tree) < 5:
                continue

            for j in range(len(tree) - 1):
                edge_list.append((point_indexs[tree[j]], point_indexs[tree[j + 1]]))

            #get crucial points
            min_tree_points = []
            length_tree = len(tree)
            for j in range(length_tree):
                min_tree_points.append(points[point_indexs[tree[j]]])
            crucial_point_indexs = douglasPeucker(min_tree_points, 0, length_tree - 1, 3 * max_edge_dist)
            size_crucial_points = len(crucial_point_indexs)
            for index in crucial_point_indexs:
                crucial_point_indexs_real.append(point_indexs[tree[index]])


        borders[i]['edges'] = edge_list
        borders[i]['crucial_points'] = crucial_point_indexs_real
    

    for i in range(len(borders)):
        edges = borders[i]['edges']
        crucial_points = borders[i]['crucial_points']
        for edge in edges:
            a, b = edge
            all_new_edges.append((a, b))
        for index in crucial_points:
            if not index in all_crucial_points:
                all_crucial_points.append(index)
            else: 
                kebab = 0
    return all_new_points, all_new_edges, all_crucial_points

        

    
    