import os
import json
import numpy as np
from plyfile import *
import glob

#文件夹名
base_dir_name = 'E:\\dataset\\scannet\\scans'
scene_name = 'scene0000_00'
dir_name = os.path.join(base_dir_name, scene_name)

instance_descriptor_name = os.path.join(dir_name, 'scene0000_00.aggregation.json') 
ply_name = os.path.join(dir_name, 'scene0000_00_vh_clean_decreased.ply')
part_dir = os.path.join(dir_name, 'instances')
with open(instance_descriptor_name, 'r', encoding = 'utf8')as fp:
    instance_info = json.load(fp)


seg_name_list = []
info_name_list = []
for filename in glob.glob(part_dir + '\\*.off'):
    seg_name = filename.split('\\')[-1]
    seg_name = seg_name.split('.')[0]
    seg_id = int(seg_name[5:])
    info_name = seg_name + '.txt'
    seg_name = seg_name + '_result.txt'
    seg_name_list.append(seg_name)
    info_name_list.append(info_name)

plydata = PlyData.read(ply_name)
vertexs = plydata['vertex']
faces = plydata['face']
face_num = faces.count
vertex_num = vertexs.count

segment = [0] * face_num
seg_num = 0




#划分每个面片的标签
for i in range(len(seg_name_list)):
    seg_name = seg_name_list[i]
    seg_name_full = os.path.join(part_dir, seg_name)
    info_name = info_name_list[i]
    info_name_full = os.path.join(part_dir, info_name)

    
    #读每个seg信息，赋值
    file = open(seg_name_full, 'r')
    data = file.readlines()
    file.close()
    for line in data:
        face_infos = line.split()
        seg_num += 1
        for face_info in face_infos:
            t_face = int(face_info)
            if segment[t_face] == 0:
                segment[t_face] = seg_num
    
    
    #读剩下的信息，赋值
    file = open(info_name_full, 'r')
    data = file.readlines()
    file.close()
    seg_num += 1
    for face_info in data:
        t_face = int(face_info[:-1])
        if segment[t_face] == 0:
            segment[t_face] = seg_num


count = 0
for i in range(len(segment)):
    if segment[i] == 0:
        count += 1
print(count)

#初始化每个点的颜色为0
vertex_r = [0] * vertex_num
vertex_g = [0] * vertex_num
vertex_b = [0] * vertex_num
vertex_faces = [0] * vertex_num




#赋值颜色
for i in range(face_num):
    seg_id = segment[i]
    r = ((seg_id // 10) * 25) % 256
    seg_redux = seg_id - (seg_id // 10) * 256
    g = ((seg_redux // 10) * 25) % 256
    b = ((seg_redux % 10) * 25) % 256
    face = faces[i][0]
    for point_id in face:
        vertex_faces[point_id] += 1
        vertex_r[point_id] += r
        vertex_g[point_id] += g
        vertex_b[point_id] += b


for i in range(vertex_num):
    if vertex_faces[i] == 0:
        continue
    vertex_r[i] = vertex_r[i] // vertex_faces[i]
    vertex_g[i]= vertex_g[i] // vertex_faces[i]
    vertex_b[i] = vertex_b[i] // vertex_faces[i]

for i in range(vertex_num):
    vertexs[i]['red'] = vertex_r[i]
    vertexs[i]['green'] = vertex_g[i]
    vertexs[i]['blue'] = vertex_b[i]


save_name = 'scene0000_00_vh_clean_segmented.ply'
save_place = os.path.join(dir_name, save_name)
plydata.write(save_place)


