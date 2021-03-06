import os
import json
from plyfile import *

#文件夹名
base_dir_name = 'E:\\dataset\\scannet\\scans'
scene_name = 'scene0000_00'
dir_name = os.path.join(base_dir_name, scene_name)

instance_descriptor_name = os.path.join(dir_name, 'scene0000_00.aggregation.json') 
ply_name = os.path.join(dir_name, 'scene0000_00_vh_clean_2.ply')
part_dir = os.path.join(dir_name, 'instances')
if not os.path.exists(part_dir):
    os.mkdir(part_dir)

with open(instance_descriptor_name, 'r', encoding = 'utf8')as fp:
    instance_info = json.load(fp)

instances = instance_info['segGroups']
vertex_info_name = os.path.join(dir_name, instance_info['segmentsFile'][8:])

plydata = PlyData.read(ply_name)
vertexs = plydata['vertex']
faces = plydata['face']

with open(vertex_info_name, 'r', encoding = 'utf8')as fp:
    vertex_info = json.load(fp)['segIndices']

print(len(vertex_info))

vertex_segments_dict = {}

for i in range(len(vertex_info)):
    item = vertex_info[i]
    if item in vertex_segments_dict:
        vertex_segments_dict[item].append(i)
    else: 
        vertex_segments_dict[item] = []
        vertex_segments_dict[item].append(i)



count = 0
for item in instances:
    vertex_group_list = item['segments']
    item_id = str(item['objectId'])
    part_name = os.path.join(part_dir, 'part_' + item_id + '.off')
    descript_name = os.path.join(part_dir, 'part_' + item_id + '.txt')
    off = open(part_name, 'w')
    txt = open(descript_name, 'w')

    point_list = []
    face_list = []
    index = [-1] * vertexs.count
    for group in vertex_group_list:
        point_list += vertex_segments_dict[group]
    
    for i in range(len(point_list)):
        index[point_list[i]] = i

    for i in range(faces.count):
        face = faces[i]
        a = face[0][0]
        b = face[0][1]
        c = face[0][2]
        if index[a] >= 0 and index[b] >= 0 and index[c] >= 0:
            place_a = index[a]
            place_b = index[b]
            place_c = index[c]
            place = [place_a, place_b, place_c]
            face_list.append(place)
            txt.write(str(i) + '\n')

    off.write('OFF\n')
    off.write(str(len(point_list)) + ' ' + str(len(face_list)) + ' ' + '0\n')
    for point in point_list:
        x = vertexs[point]['x']
        y = vertexs[point]['y']
        z = vertexs[point]['z']
        off.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
    for face in face_list:
        off.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')
    off.close()
    txt.close()
    print("written part", item_id, "total", len(point_list), "points", len(face_list), 'faces')
print(count)
save_name = 'scene0000_00_vh_clean_test.ply'
save_place = os.path.join(dir_name, save_name)
plydata.write(save_place)