import h5py
import numpy as np
import os
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import skimage.io as io


class MatterPortDataSet(Dataset):
    def __init__(self, base_dir, the_type):
        '''
        description: init the dataset
        parameter: the base dir of the dataset, the type(training, validation, testing)
        return:empty
        '''
        self.base_dir = base_dir 
        self.type = the_type
        self.mat_path = os.path.join(base_dir, the_type, the_type + '.mat')
        self.depth_filenames = []
        self.image_filenames = []
        self.init_label_filenames = []
        self.layout_depth_filenames = []
        self.layout_seg_filenames = []
        self.points = []
        self.intrinsics = []
        with h5py.File(self.mat_path, 'r') as f:
            data = f['data']
            depths = data['depth'][:]
            images = data['image'][:]
            init_labels = data['init_label'][:]
            intrinsics_matrixs = data['intrinsics_matrix'][:]
            layout_depths = data['layout_depth'][:]
            layout_segs = data['layout_seg'][:]
            models = data['model'][:]
            points = data['point'][:]
            self.length = len(depths)
            for i in range(self.length):
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
                the_model_faces = the_model['face']
                the_model_params = the_model['params']

                self.faces = []
                self.params = []
                for j in range(len(the_model_faces)):
                    face = f[the_model_faces[j][0]][0][0]
                    param = f[the_model_params[j][0]][0][0]
                    self.faces.append(face) 
                    self.params.append(param) 
                
                self.intrinsics.append(the_intrinsic)
                self.points.append(the_point)

                the_string = ''
                for item in depth_id: 
                    the_string += chr(item[0])
                self.depth_filenames.append(the_string)

                the_string = ''
                for item in image_id: 
                    the_string += chr(item[0])
                self.image_filenames.append(the_string)

                the_string = ''
                for item in init_label_id: 
                    the_string += chr(item[0])
                self.init_label_filenames.append(the_string)

                the_string = ''
                for item in layout_depth_id: 
                    the_string += chr(item[0])
                self.layout_depth_filenames.append(the_string)

                the_string = ''
                for item in layout_seg_id: 
                    the_string += chr(item[0])
                self.layout_seg_filenames.append(the_string)

        self.depths = [] 
        self.images = [] 
        self.init_labels = [] 
        self.layout_depths = []
        self.layout_segs = []
        for i in range(self.length):
            depth_name = os.path.join(self.base_dir, self.type, 'depth', self.depth_filenames[i])
            image_name = os.path.join(self.base_dir, self.type, 'image', self.image_filenames[i])
            init_label_name = os.path.join(self.base_dir, self.type, 'init_label', self.init_label_filenames[i])
            layout_depth_name = os.path.join(self.base_dir, self.type, 'layout_depth', self.layout_depth_filenames[i])
            layout_seg_name = os.path.join(self.base_dir, self.type, 'layout_seg', self.layout_seg_filenames[i])

            depth = io.imread(depth_name)
            image = io.imread(image_name)
            init_label = io.imread(init_label_name)
            layout_depth = io.imread(layout_depth_name)
            layout_seg = io.imread(layout_seg_name)

            self.depths.append(depth)
            self.images.append(image)
            self.init_labels.append(init_label)
            self.layout_depths.append(layout_depth)
            self.layout_segs.append(layout_seg)
 
    def __getitem__(self, i):
        '''
        description: get one part of the item
        parameter: the index 
        return: the data
        '''
        return self.depths[i], self.images[i], self.init_labels[i], self.layout_depths[i], self.layout_segs[i], \
            self.faces[i], self.params, self.intrinsics[i], self.points[i]
 
    def __len__(self):
        '''
        description: get the length of the dataset
        parameter: empty
        return: the length
        '''
        return self.length



a = MatterPortDataSet('E:\\dataset\\geolayout', 'validation')
print(a.__len())
depth, image, init_label, layout_depth, layout_seg, face, param, intrinsic, point = a.__getitem__(10)
print(depth)
print(image)
print(init_label)
print(layout_depth)
print(layout_seg)
print(face)
print(param)
print(intrinsic)
print(point)