import numpy as np
import cv2
import sys
import os
from scipy.optimize import leastsq
from plyfile import PlyData, PlyElement
import json
import glob


def getPlaneIndex(lineIndex, planes):
    return lineIndex // planes, lineIndex % planes

def getLineIndex(planeIndexA, planeIndexB, planes):
    mini = min(planeIndexA, planeIndexB)
    maxi = max(planeIndexA, planeIndexB)
    return mini * planes + maxi




def residuals(p, X, Y):
    loss = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        loss += (y - p[0] * x[0] - p[1] * x[1] - p[2] * x[2] - p[3]) ** 2
    return loss


def fit_line(points):
    Y = np.zeros(points.shape[0])
    pars = np.random.rand(4) 
    r = leastsq(residuals, pars, args = (points, Y))   # 三个参数：误差函数、函数参数列表、数据点
    return r



def getBorder(planes, planeSegmentation, points, faces):
    '''
    description: After get the points of all segmented planes, get the border lines(both points and geometry approximation)
    parameter: plane_segmentation, points, faces
    return: border points, border line parameter
    '''
    lines = []
    useful_line_indexs = []
    lines_point = []
    lines_parameter = []
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
    #regress all lines
    for i in range(len(lines_point)):
        points = lines_point[i]
        lines_parameter.append(fit_line(points)[0])
    lines_parameter = np.array(lines_parameter)


    return lines_point, lines_parameter

        

    
    