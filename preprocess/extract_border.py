import numpy as np
import cv2
import sys
import os
from scipy.optimize import leastsq
from plyfile import PlyData, PlyElement
import json
import glob
from math import *

def getPlaneIndex(lineIndex, planes):
    return lineIndex // planes, lineIndex % planes

def getLineIndex(planeIndexA, planeIndexB, planes):
    mini = min(planeIndexA, planeIndexB)
    maxi = max(planeIndexA, planeIndexB)
    return mini * planes + maxi

def getDist(a, b):
    return sqrt(np.sum((a - b) ** 2, axis = 0))

def getMinTree(points):
    '''
    description: get the min generated tree of the points
    parameter: points
    return: the tree
    '''
    n = points.shape[0]
    edges = np.zeros((n, n))
    visit = [False] * n
    v_new = []
    e_new = []
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
        elif min_u == v_new[-1]:
            v_new.append(min_v)
    return v_new




def getBorder(planes, planeSegmentation, points, faces):
    '''
    description: After get the points of all segmented planes, get the border lines(both points and geometry approximation)
    parameter: plane_segmentation, points, faces
    return: border points, border line parameter
    '''
    lines = []
    useful_line_indexs = []
    lines_point = []
    lines_result = []
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
            useful_line_indexs.append(getPlaneIndex(i, planes))
            new_point_list = []
            for index_a, index_b in lines[i]:
                point_a = points[index_a]
                point_b = points[index_b]
                point_middle = (point_a + point_b) / 2
                new_point_list.append(point_middle)
            new_point_list = np.array(new_point_list)
            lines_point.append(new_point_list)

    lines_point = np.array(lines_point)
    for points in lines_point:
        min_tree = getMinTree(points)
        edge_list = []
        for i in range(len(min_tree) - 1):
            edge_list.append((min_tree[i], min_tree[i + 1]))
        lines_result.append(edge_list)
    
    all_new_points = []
    all_new_edges = []
    for i in range(len(lines_point)):
        points = lines_point[i]
        edges = lines_result[i]
        current_point_num_base = len(all_new_points)
        for point in points:
            all_new_points.append(point)
        for edge in edges:
            a, b = edge
            new_a = a + current_point_num_base
            new_b = b + current_point_num_base
            all_new_edges.append((new_a, new_b))

    return all_new_points, all_new_edges

        

    
    