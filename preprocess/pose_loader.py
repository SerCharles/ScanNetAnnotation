import numpy as np
import skimage.io as sio
import glob
import os
import functools

def compareFilename(name_a, name_b):
    id_a = int(name_a.split(os.sep)[-1][:-4])
    id_b = int(name_b.split(os.sep)[-1][:-4])
    return id_a - id_b 


def loadPoses(pose_path, interval):
    '''
    description: load the pose files to be rendered
    input: pose_path, render interval
    return: the list of poses
    '''
    pose_files = glob.glob(os.path.join(pose_path, '*.txt'))
    pose_files.sort(key = functools.cmp_to_key(compareFilename))
    poses = []
    for i in range(len(pose_files)):
        if i % interval == 0:
            pose = []
            file_name = pose_files[i]
            lines = [l.strip() for l in open(pose_files[i])]
            for j in range(len(lines)):
                l = lines[j]
                words = l.split()
                if len(words) == 0:
                    continue
                pose.append([])
                for word in words:
                    word = float(word)
                    pose[j].append(word)
            pose = np.array(pose, dtype='float32')
            poses.append(pose)
    return poses

def leavePictures(picture_path, interval):
    '''
    description: only leave the relevant picturs
    input: picture path, render interval
    return: empty
    '''
    picture_files = glob.glob(os.path.join(picture_path, '*.jpg'))
    picture_files.sort(key = functools.cmp_to_key(compareFilename))
    for i in range(len(picture_files)):
        p_id = int(picture_files[i].split(os.sep)[-1][:-4])
        if p_id % interval != 0:
            file_name = picture_files[i]
            os.remove(file_name)

def getGLParameters(ROOT_FOLDER, scene_id):
    '''
    description: load the pose files and switch them to the eye, center and poses that opengl need
    input: root folder, scene id
    return: empty, write the file
    '''
    pose_path = os.path.join(ROOT_FOLDER, scene_id, 'pose')
    picture_path = os.path.join(ROOT_FOLDER, scene_id, 'color')
    save_file = os.path.join(ROOT_FOLDER, scene_id, 'annotation', 'poses.txt')
    interval = 30
    poses = loadPoses(pose_path, interval)
    leavePictures(picture_path, interval)
    f = open(save_file, 'w')
    f.write(str(len(poses)) + '\n')
    for ii in range(len(poses)):
        pose = poses[ii]
        pose = np.linalg.inv(pose)
        pose_id = interval * ii
        x = np.zeros(3) 
        y = np.zeros(3) 
        z = np.zeros(3) 
        result_move = np.zeros(3) 
        for i in range(3):
            x[i] = pose[0][i] 
            y[i] = pose[1][i] 
            z[i] = pose[2][i]
            result_move[i] = pose[i][3]
        r = np.concatenate((x, y, z)).reshape((3, 3))
        center = - np.matmul(np.linalg.inv(r), result_move)
        eye = center - z 
        up = - np.cross(z, x)
        f.write(str(pose_id) + '\n')
        f.write(str(eye[0]) + ' ' + str(eye[1]) + ' ' + str(eye[2]) + '\n')
        f.write(str(center[0]) + ' ' + str(center[1]) + ' ' + str(center[2]) + '\n')
        f.write(str(up[0]) + ' ' + str(up[1]) + ' ' + str(up[2]) + '\n')
    f.close()


ROOT_FOLDER = "E:\\dataset\\scannet\\scans"
scene_id = 'scene0000_00'
getGLParameters(ROOT_FOLDER, scene_id)
