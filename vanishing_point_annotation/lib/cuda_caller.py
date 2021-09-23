from ctypes import *
import numpy as np
import os

libpath = os.path.dirname(os.path.abspath(__file__))
print(libpath)
Annotator = cdll.LoadLibrary(os.path.join(libpath, 'Annotator.so'))

def get_ceiling_and_floor(layout_seg, lines, ceiling_id, floor_id):
    """Get the ceiling place and floor place of each line from the vanishing point to the picture, and also whether there are ceiling and floor 
        H: the height of the picture
        W: the width of the picture

    Args:
        layout_seg [numpy int array], [H * W]: [the layout segmentation of the picture]
        lines [float array], [(2 * W) * 2]: [the sampled lines, the four instances are top_x, bottom_x]
        ceiling_id [int]: [the id of the plane which is the ceiling]
        floor_id [int]: [the id of the plane which is the floor]

    Return:
        whether_ceilings [numpy boolean array], [(2 * W)]: [whether the lines have ceiling]
        whether_floors [numpy boolean array], [(2 * W)]: [whether the lines have floor]
        whether_walls [numpy boolean array], [(2 * W)]: [whether the lines have wall]
        ceiling_places [numpy float array], [2 * (2 * W)]: [the ceiling place of each line, (y, x)]
        floor_places [numpy float array], [2 * (2 * W)]: [the floor place of each line, (y, x)]
    """
    H, W = layout_seg.shape
    line_bottom_x = np.zeros((2 * W), dtype="float32")
    line_top_x = np.zeros((2 * W), dtype="float32")
    layout_seg = layout_seg.reshape(H * W).astype("int32")
    whether_ceilings = np.zeros((2 * W), dtype="bool")
    whether_floors = np.zeros((2 * W), dtype="bool")
    whether_walls = np.zeros((2 * W), dtype="bool")
    ceilings_y = np.zeros((2 * W), dtype="float32")
    ceilings_x = np.zeros((2 * W), dtype="float32")
    floors_y = np.zeros((2 * W), dtype="float32")
    floors_x = np.zeros((2 * W), dtype="float32")
    
    for i in range(2 * W):
        line_top_x[i] = lines[i][0]
        line_bottom_x[i] = lines[i][1]

    c_H = c_int(H)
    c_W = c_int(W)
    c_floor_id = c_int(floor_id)
    c_ceiling_id = c_int(ceiling_id)
    c_line_bottom_x = c_void_p(line_bottom_x.ctypes.data)
    c_line_top_x = c_void_p(line_top_x.ctypes.data)
    c_layout_seg = c_void_p(layout_seg.ctypes.data)
    c_whether_ceilings = c_void_p(whether_ceilings.ctypes.data)
    c_whether_walls = c_void_p(whether_walls.ctypes.data)
    c_whether_floors = c_void_p(whether_floors.ctypes.data)
    c_ceilings_y = c_void_p(ceilings_y.ctypes.data)
    c_ceilings_x = c_void_p(ceilings_x.ctypes.data)
    c_floors_y = c_void_p(floors_y.ctypes.data)
    c_floors_x = c_void_p(floors_x.ctypes.data)

    Annotator.run(c_H, c_W, c_floor_id, c_ceiling_id, c_line_bottom_x, c_line_top_x, c_layout_seg, \
        c_whether_ceilings, c_whether_walls, c_whether_floors, c_ceilings_y, c_ceilings_x, c_floors_y, c_floors_x)
    
    ceiling_places = np.stack((ceilings_y, ceilings_x), axis=0)
    floor_places = np.stack((floors_y, floors_x), axis=0)


    return whether_ceilings, whether_floors, whether_walls, ceiling_places, floor_places

    
    
    