import numpy as np
from math import *

def getDistance(p, start, end):
    '''
    description: get the distance between a point and the start and end of a line
    parameter: point p, line start, line end
    return: distance
    '''
    x, y, z = p
    x1, y1, z1 = start 
    x2, y2, z2 = end 
    m = x2 - x1 
    n = y2 - y1 
    p = z2 - z1 
    t = ((x - x1) * m + (y - y1) * n + (z - z1) * p)/(m ** 2 + n ** 2 + p ** 2)
    xx = x1 + m * t
    yy = y1 + n * t 
    zz = z1 + p * t 
    dist = sqrt((x - xx) ** 2 + (y - yy) ** 2 + (z - zz) ** 2)
    return dist


def douglasPeucker(point_list, start_id, end_id, threshold):
    '''
    description: Find the point with the maximum distance
    parameter: points that form a border line, start_index, end_index, the threshold
    return: the indexes of new point list
    '''
    max_dist = 0
    max_index = 0
    for i in range(start_id + 1, end_id + 1):
        dist = getDistance(point_list[i], point_list[start_id], point_list[end_id])
        if (dist > max_dist):
            max_index = i
            max_dist = dist
        
    result_list = []
    
    #If max distance is greater than epsilon, recursively simplify
    if (max_dist > threshold):
        #Recursive call
        rec_results1 = douglasPeucker(point_list, start_id, max_index, threshold)
        rec_results2 = douglasPeucker(point_list, max_index, end_id, threshold)
        result_list += rec_results1
        result_list += rec_results2
    else:
        return range(start_id, end_id + 1)
    return result_list