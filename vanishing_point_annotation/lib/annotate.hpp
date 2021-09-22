#ifndef ANNOTATE_H_
#define ANNOTATE_H_


void Annotate(int H, int W, int floor_id, int ceiling_id, float* line_bottom_x, float* line_top_x, int* layout_seg, 
bool* whether_ceilings, bool* whether_walls, bool* whether_floors, float* ceilings_y, float* ceilings_x, float* floors_y, float* floors_x);

#endif