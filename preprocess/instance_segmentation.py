import os
import json
import numpy as np
from plyfile import *

type_use = ['window', 'kitchen counter', 'cabinet', \
    'kitchen cabinets', 'floor', 'wall', 'door', 'ceiling', 'doorframe', 'shelf']

'''
描述：选取所需的模型，并且更新各种数据文件
参数：根目录，场景id
返回：无
'''
def select_shape(ROOT_FOLDER, scene_id):
    ply_name = os.path.join(ROOT_FOLDER, scene_id, scene_id + '_vh_clean_2.labels.ply')
    instance_descriptor_name = os.path.join(ROOT_FOLDER, scene_id, scene_id + '.aggregation.json') 
    vertex_info_name = os.path.join(ROOT_FOLDER, scene_id, scene_id + '_vh_clean_2.0.010000.segs.json')

    plydata = PlyData.read(ply_name)
    vertexs = plydata['vertex']
    faces = plydata['face']

    with open(instance_descriptor_name, 'r', encoding = 'utf8')as fp:
        instance_info = json.load(fp)
    instances = instance_info['segGroups']

    with open(vertex_info_name, 'r', encoding = 'utf8')as fp:
        vertex_info_total = json.load(fp)
        vertex_info = vertex_info_total['segIndices']

    vertex_segments_dict = {}

    for i in range(len(vertex_info)):
        item = vertex_info[i]
        if item in vertex_segments_dict:
            vertex_segments_dict[item].append(i)
        else: 
            vertex_segments_dict[item] = []
            vertex_segments_dict[item].append(i)




    #原先的第i个点现在变成index_total个点了
    index_total = [-1] * vertexs.count
    total_useful_points = 0
    show_vertexs = []


    for item in instances:
        vertex_group_list = item['segments']
        item_id = item['objectId']
        item_type = item['label']
        if not item_type in type_use:
            continue
        previous_point_list = []
        for group in vertex_group_list:
            previous_point_list += vertex_segments_dict[group]
        for i in range(len(previous_point_list)):
            index_total[previous_point_list[i]] = total_useful_points
            total_useful_points += 1
            show_vertexs.append(vertexs[previous_point_list[i]])


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
 
    show_vertexs = np.array(show_vertexs)
    show_faces = np.array(show_faces, dtype=[('vertex_indices', 'uint32', (3,))])
    new_vertex = PlyElement.describe(show_vertexs, 'vertex')
    new_face = PlyElement.describe(show_faces, 'face')
    show_ply = PlyData([new_vertex, new_face], text = True, byte_order = '<')

    save_place = os.path.join(ROOT_FOLDER, scene_id, scene_id + '_vh_clean_2_selected.ply')
    show_ply.write(save_place)
    
    new_segment_info = [-1] * len(show_vertexs)
    segment_dict = {}
    for i in range(len(vertex_info)):
        if index_total[i] >= 0:
            new_index = index_total[i]
            old_segment = vertex_info[i]
            new_segment = index_total[old_segment]
            new_segment_info[new_index] = new_segment
            segment_dict[old_segment] = new_segment

    vertex_info_total['segIndices'] = new_segment_info

    new_json_segment_name = os.path.join(ROOT_FOLDER, scene_id, scene_id + '_vh_clean_2.0.010000_selected.segs.json')
    with open(new_json_segment_name, "w") as f:
        json.dump(vertex_info_total, f)

    new_instances = []
    for item in instances:
        if item['label'] in type_use:
            for i in range(len(item['segments'])):
                old_seg_id = item['segments'][i]
                item['segments'][i] = segment_dict[old_seg_id]
            new_instances.append(item)
    
    instance_info['segGroups'] = new_instances
    new_json_instance_name = os.path.join(ROOT_FOLDER, scene_id, scene_id + '_selected.aggregation.json') 
    with open(new_json_instance_name, "w") as f:
        json.dump(instance_info, f)
    
