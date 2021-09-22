#include <iostream>
#include <cstring>
#include "annotate.hpp"


extern "C" {

void run(int H, int W, int floor_id, int ceiling_id, float* line_bottom_x, float* line_top_x, int* layout_seg, 
    bool* whether_ceilings, bool* whether_walls, bool* whether_floors, float* ceilings_y, float* ceilings_x, float* floors_y, float* floors_x)
{
    Annotate(H, W, floor_id, ceiling_id, line_bottom_x, line_top_x, layout_seg, whether_ceilings, whether_walls, whether_floors,
        ceilings_y, ceilings_x, floors_y, floors_x);
}








}