import os
import json
from plyfile import *
import glob

#文件夹名
base_dir_name = 'E:\\dataset\\scannet\\scans'
scene_name = 'scene0000_00'
dir_name = os.path.join(base_dir_name, scene_name)

instance_descriptor_name = os.path.join(dir_name, 'scene0000_00.aggregation.json') 
ply_name = os.path.join(dir_name, 'scene0000_00_vh_clean_3.ply')
part_dir = os.path.join(dir_name, 'instances')

seg_name_list = []
point_name_list = []

for filename in glob.glob(part_dir + '\\*.pwn'):
    seg_name = filename.split('\\')[-1]
    seg_name = seg_name.split('.')[0]
    point_name = seg_name + '.txt'
    seg_name = seg_name + '_result.txt'
    seg_name_list.append(seg_name)
    point_name_list.append(point_name)

plydata = PlyData.read(ply_name)
vertexs = plydata['vertex']
faces = plydata['face']
point_num = vertexs.count

segment = [0] * point_num
seg_num = 0

for i in range(len(seg_name_list)):
    seg_name = seg_name_list[i]
    point_name = point_name_list[i]
    seg_name_full = os.path.join(part_dir, seg_name)
    point_name_full = os.path.join(part_dir, point_name)

    #读每个seg信息，赋值
    file = open(seg_name_full, 'r')
    data = file.readlines()
    file.close()
    for line in data:
        points = line.split()
        seg_num += 1
        for point in points:
            t_point = int(point)
            if segment[t_point] == 0:
                segment[t_point] = seg_num
    
    #读剩下的信息，赋值
    file = open(point_name_full, 'r')
    points = file.readlines()
    file.close()
    seg_num += 1
    for point in points:
        t_point = int(point[:-1])
        if segment[t_point] == 0:
            segment[t_point] = seg_num

print(segment)
for i in range(vertexs.count):
    seg_id = segment[i]
    r = ((seg_id // 256) * 16) % 256
    seg_redux = seg_id - (seg_id // 256) * 256
    g = ((seg_redux // 16) * 16) % 256
    b = ((seg_redux % 16) * 16) % 256
    vertexs[i]['red'] = r
    vertexs[i]['green'] = g
    vertexs[i]['blue'] = b

save_name = 'scene0000_00_vh_clean_segmented.ply'
save_place = os.path.join(dir_name, save_name)
plydata.write(save_place)


