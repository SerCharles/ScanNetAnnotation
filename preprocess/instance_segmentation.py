import os
import json
from plyfile import *

#文件夹名
base_dir_name = 'E:\\dataset\\scannet\\scans'
scene_name = 'scene0000_00'
dir_name = os.path.join(base_dir_name, scene_name)

instance_descriptor_name = os.path.join(dir_name, 'scene0000_00.aggregation.json') 
ply_name = os.path.join(dir_name, 'scene0000_00_vh_clean.ply')
part_dir = os.path.join(dir_name, 'instances')
if not os.path.exists(part_dir):
    os.mkdir(part_dir)

with open(instance_descriptor_name, 'r', encoding = 'utf8')as fp:
    instance_info = json.load(fp)

instances = instance_info['segGroups']
vertex_info_name = os.path.join(dir_name, instance_info['segmentsFile'][8:])

plydata = PlyData.read(ply_name)
vertexs = plydata['vertex']

with open(vertex_info_name, 'r', encoding = 'utf8')as fp:
    vertex_info = json.load(fp)['segIndices']


vertex_segments_dict = {}

for i in range(len(vertex_info)):
    item = vertex_info[i]
    if item in vertex_segments_dict:
        vertex_segments_dict[item].append(i)
    else: 
        vertex_segments_dict[item] = []


for item in instances:
    vertex_group_list = item['segments']
    item_id = str(item['objectId'])
    part_name = os.path.join(part_dir, 'part_' + item_id + '.pwn')
    descript_name = os.path.join(part_dir, 'part_' + item_id + '.txt')
    
    pwn = open(part_name, 'w')
    txt = open(descript_name, 'w')
    point_num = 0
    for group in vertex_group_list:
        point_list = vertex_segments_dict[group]
        point_num += len(point_list)
        for i in point_list:
            x = vertexs[i]['x']
            y = vertexs[i]['y']
            z = vertexs[i]['z']
            nx = vertexs[i]['nx']
            ny = vertexs[i]['ny']
            nz = vertexs[i]['nz']
            pwn.write(str(x) + ' ')
            pwn.write(str(y) + ' ')
            pwn.write(str(z) + ' ')
            pwn.write(str(nx) + ' ')
            pwn.write(str(ny) + ' ')
            pwn.write(str(nz))
            pwn.write('\n')
            txt.write(str(i) + '\n')
    pwn.close()
    txt.close()
    print("written part", item_id, "total", point_num, "points")