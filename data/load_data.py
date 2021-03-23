import h5py
import numpy as np
path = 'E:\\dataset\\geolayout\\training\\training.mat'


depth_filenames = []
image_filenames = []
init_label_filenames = []
layout_depth_filenames = []
layout_seg_filenames = []
model_data = []
point_data = []
intrinsic_data = []
with h5py.File(path, 'r') as f:
    data = f['data']
    depths = data['depth'][:]
    images = data['image'][:]
    init_labels = data['init_label'][:]
    intrinsics_matrixs = data['intrinsics_matrix'][:]
    layout_depths = data['layout_depth'][:]
    layout_segs = data['layout_seg'][:]
    models = data['model'][:]
    points = data['point'][:]
    for i in range(len(depths)):
        depth_id = f[depths[i][0]]
        image_id = f[images[i][0]]
        init_label_id = f[init_labels[i][0]]
        layout_depth_id = f[layout_depths[i][0]]
        layout_seg_id = f[layout_segs[i][0]]
        the_intrinsic = f[intrinsics_matrixs[i][0]]
        the_model = f[models[i][0]]
        the_point = f[points[i][0]]

        the_intrinsic = np.array(the_intrinsic)
        the_point = np.array(the_point)
        the_model_face = the_model['face']
        the_model_params = the_model['params']

        face_0 = f[the_model_face[0][0]][0][0]
        face_1 = f[the_model_face[1][0]][0][0]
        face_2 = f[the_model_face[2][0]][0][0]
        param_0 = f[the_model_params[0][0]][0][0]
        param_1 = f[the_model_params[1][0]][0][0]
        param_2 = f[the_model_params[2][0]][0][0]
        model_data.append([[face_0, face_1, face_2], [param_0, param_1, param_2]])

        
        the_string = ''
        for item in depth_id: 
            the_string += chr(item[0])
        depth_filenames.append(the_string)

        the_string = ''
        for item in image_id: 
            the_string += chr(item[0])
        image_filenames.append(the_string)

        the_string = ''
        for item in init_label_id: 
            the_string += chr(item[0])
        init_label_filenames.append(the_string)

        the_string = ''
        for item in layout_depth_id: 
            the_string += chr(item[0])
        layout_depth_filenames.append(the_string)

        the_string = ''
        for item in layout_seg_id: 
            the_string += chr(item[0])
        layout_seg_filenames.append(the_string)

        
