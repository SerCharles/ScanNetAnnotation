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




def residuals(p, points):
    '''
    description: get the residuals, used in line fitting
    parameters: params(a, b, x0, y0); points(x, y, z)
    return: loss, length is 2n 
    '''
    n = points.shape[0]
    loss = np.zeros(2 * n)
    a, b, x0, y0 = p
    for i in range(n):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]

        loss_1 = a * z + x0 - x
        loss_2 = b * z + y0 - y
        loss[2 * i] = loss_1
        loss[2 * i + 1] = loss_2
    return loss



def fit_line(points):
    '''
    description: fit the line based on all points
    parameters:points(x, y, z)
    return: (m, n, p, x0, y0, z0) 
    s.t. (x, y, z) = (mt + x0, nt + y0, pt + z0)
    '''
    n = points.shape[0]
    
    if n == 1:
        return np.array([0, 0, 0, points[0][0], points[0][1], points[0][2]])
    
    #par: a, b, x0, y0
    #x = x0 + az, y = y0 + bz
    pars = np.random.rand(4) 
    new_pars = leastsq(residuals, pars, args = (points))[0]
    a, b, x0, y0 = new_pars
    return np.array([a, b, 1, x0, y0, 0])




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
        lines_parameter.append(fit_line(points))
    lines_parameter = np.array(lines_parameter)


    return lines_point, lines_parameter

        

    
    