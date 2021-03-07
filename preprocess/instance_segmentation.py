import os
import json
import numpy as np
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


type_use = ['window', 'table', 'kitchen counter', 'desk', 'cabinet', 'floor', 'wall', 'coffee table', 'door', 'ceiling', 'shelf', 'doorframe']
index_total = [-1] * vertexs.count
total_useful_points = 0
show_vertexs = []

point_count = 0
face_count = 0
point_list_total = {}


for item in instances:
    vertex_group_list = item['segments']
    item_id = item['objectId']
    item_type = item['label']
    if not item_type in type_use:
        continue
    point_list_total[item_id] = []
    point_list = []
    for group in vertex_group_list:
        point_list += vertex_segments_dict[group]
        point_count += len(vertex_segments_dict[group])
    for i in range(len(point_list)):
        index_total[point_list[i]] = total_useful_points
        point_list_total[item_id].append(total_useful_points)
        total_useful_points += 1
        show_vertexs.append(vertexs[point_list[i]])
print(point_count, face_count)


show_faces = []
for i in range(faces.count):
    face = faces[i]
    a = face[0][0]
    b = face[0][1]
    c = face[0][2]
    if index_total[a] >= 0 and index_total[b] >= 0 and index_total[c] >= 0:
        place_a = index_total[a]
        place_b = index_total[b]
        place_c = index_total[c]
        place = ([place_a, place_b, place_c],)
        show_faces.append(place)
print(len(show_vertexs), len(show_faces))
 
show_vertexs = np.array(show_vertexs)
show_faces = np.array(show_faces, dtype=[('vertex_indices', 'uint32', (3,))])
new_vertex = PlyElement.describe(show_vertexs, 'vertex')
new_face = PlyElement.describe(show_faces, 'face')
show_ply = PlyData([new_vertex, new_face], text = True, byte_order = '<')

save_name = 'scene0000_00_vh_clean_decreased.ply'
save_place = os.path.join(dir_name, save_name)
show_ply.write(save_place)

vertexs = show_ply['vertex']
faces = show_ply['face']
for (item_id, point_list) in point_list_total.items():
    part_name = os.path.join(part_dir, 'part_' + str(item_id) + '.off')
    descript_name = os.path.join(part_dir, 'part_' + str(item_id) + '.txt')
    off = open(part_name, 'w')
    txt = open(descript_name, 'w')

    face_list = []
    index = [-1] * vertexs.count
    #print(vertexs.count)
    #print(len(point_list))
    for i in range(len(point_list)):
        aa = point_list[i]
        index[aa] = i

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
            face_count += 1
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
    