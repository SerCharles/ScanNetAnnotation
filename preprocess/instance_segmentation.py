import os
import json
from plyfile import *

dir_name = 'E:\\dataset\\scannet\\scans\\scene0000_00'
instance_descriptor_name = os.path.join(dir_name, 'scene0000_00.aggregation.json')
ply_name = os.path.join(dir_name, 'scene0000_00_vh_clean.ply')
plydata = PlyData.read(ply_name)
vertexs = plydata['vertex']
#print(vertexs)
part_dir = os.path.join(dir_name, 'instances')

with open(instance_descriptor_name, 'r', encoding = 'utf8')as fp:
    json_data = json.load(fp)

instances = json_data['segGroups']
for item in instances:
    point_list = item['segments']
    the_id = str(item['objectId'])
    part_name = os.path.join(part_dir, 'part_' + the_id + '.pwn')
    descript_name = os.path.join(part_dir, 'part_' + the_id + '.txt')
    
    pwn = open(part_name, 'w')
    txt = open(descript_name, 'w')
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
    print("written part", the_id)