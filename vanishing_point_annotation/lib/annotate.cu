#include "annotate.hpp"
#include <iostream>
#include <stdio.h>


__constant__ int height; // the height of the picture
__constant__ int width; // the width of the picture
__constant__ int floor_label;  //the label of the floor in the layout segmentation
__constant__ int ceiling_label; //the label of the ceiling in the layout segmentation



__global__ void Annotate_gpu(float* line_bottom_x, float* line_top_x, int* layout_seg, 
bool* whether_ceilings, bool* whether_walls, bool* whether_floors, float* ceilings_y, float* ceilings_x, float* floors_y, float* floors_x)
/*The cuda function of annotating one line
    Args:
        line_bottom_x [cuda float array], [2 * W]: [the bottom x of the 2W lines]
        line_top_x [cuda float array], [2 * W]: [the top x of the 2W lines]
        layout_seg [cuda int array], [H * W]: [the layout segmentation of the picture]

    Returns:
        whether_ceilings [cuda boolean array], [2 * W]: [whether the lines have ceiling]
        whether_walls [cuda boolean array], [2 * W]: [whether the lines have wall]
        whether_floors [cuda boolean array], [2 * W]: [whether the lines have floor]
        ceilings_y [cuda float array], [2 * W]: [the ceiling y place of each line]
        ceilings_x [cuda float array], [2 * W]: [the ceiling x place of each line]
        floors_y [cuda float array], [2 * W]: [the floor y place of each line]
        floors_x [cuda float array], [2 * W]: [the floor x place of each line]
*/
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= 2 * width)
		return;

    float top_x = line_top_x[i];
    float top_y = 0.0;
    float bottom_x = line_bottom_x[i];
    float bottom_y = 0.0;
    float dy = 1.0;
    float dx = (bottom_x - top_x) / bottom_y;
    bool whether_ceiling = 0;
    bool whether_wall = 0;
    bool whether_floor = 0;
    float ceiling_x = top_x;
    float ceiling_y = top_y;
    float floor_x = bottom_x;
    float floor_y = bottom_y;
    float current_x = top_x;
    float current_y = top_y;

    for(int j = 0; j < height; j ++)
    {
        int axis_x = (int)current_x;
        int axis_y = (int)current_y;
        if(axis_x < 0 || axis_x >= width || axis_y < 0 || axis_y >= height || layout_seg[axis_y * width + axis_x] <= 0)
        {

        }
        else if(layout_seg[axis_y * width + axis_x] == ceiling_label)
        {
            whether_ceiling = 1;
            if(current_y > ceiling_y)
            {
                ceiling_x = current_x;
                ceiling_y = current_y;
            }
        }
        else if(layout_seg[axis_y * width + axis_x] == floor_label)
        {
            whether_floor = 1;
            if(current_y < floor_y)
            {
                floor_x = current_x;
                floor_y = current_y;
            }
        }
        else
        {
            whether_wall = 1;
        }
        current_x += dx;
        current_y += dy;
    }

    whether_ceilings[i] = whether_ceiling;
    whether_floors[i] = whether_floor;
    whether_walls[i] = whether_wall;
    ceilings_y[i] = ceiling_y;
    ceilings_x[i] = ceiling_x;
    floors_y[i] = floor_y;
    floors_x[i] = floor_x;    
}

void Annotate(int H, int W, int floor_id, int ceiling_id, float* line_bottom_x, float* line_top_x, int* layout_seg, 
bool* whether_ceilings, bool* whether_walls, bool* whether_floors, float* ceilings_y, float* ceilings_x, float* floors_y, float* floors_x)
/*The main function of annotating lines
    Args:
        H [int]: the height of the picture
        W [int]: [the width of the picture]
        floor_id [int]: [the id of the plane which is the floor]
        ceiling_id [int]: [the id of the plane which is the ceiling]
        line_bottom_x [float array], [2 * W]: [the bottom x of the 2W lines]
        line_top_x [float array], [2 * W]: [the top x of the 2W lines]
        layout_seg [int array], [H * W]: [the layout segmentation of the picture]

    Returns:
        whether_ceilings [boolean array], [2 * W]: [whether the lines have ceiling]
        whether_walls [boolean array], [2 * W]: [whether the lines have wall]
        whether_floors [boolean array], [2 * W]: [whether the lines have floor]
        ceilings_y [float array], [2 * W]: [the ceiling y place of each line]
        ceilings_x [float array], [2 * W]: [the ceiling x place of each line]
        floors_y [float array], [2 * W]: [the floor y place of each line]
        floors_x [float array], [2 * W]: [the floor x place of each line]
*/
{
    //init cuda memory
    cudaMemcpyToSymbol(height, &H, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(width, &W, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(floor_label, &floor_id, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(ceiling_label, &ceiling_id, sizeof(int), 0, cudaMemcpyHostToDevice);

	float* line_bottom_x_gpu;
	cudaMalloc((void**)&line_bottom_x_gpu, 2 * W * sizeof(float));
    cudaMemcpy((void*)line_bottom_x_gpu, (void*)line_bottom_x, 2 * W * sizeof(float), cudaMemcpyHostToDevice);
	
    float* line_top_x_gpu;
	cudaMalloc((void**)&line_top_x_gpu, 2 * W * sizeof(float));
    cudaMemcpy((void*)line_top_x_gpu, (void*)line_top_x, 2 * W * sizeof(float), cudaMemcpyHostToDevice);

    int* layout_seg_gpu;
	cudaMalloc((void**)&layout_seg_gpu, H * W * sizeof(int));
    cudaMemcpy((void*)layout_seg_gpu, (void*)layout_seg, H * W * sizeof(int), cudaMemcpyHostToDevice);

    bool* whether_ceilings_gpu;
    cudaMalloc((void**)&whether_ceilings_gpu, 2 * W * sizeof(bool));
    cudaMemcpy((void*)whether_ceilings_gpu, (void*)whether_ceilings, 2 * W * sizeof(bool), cudaMemcpyHostToDevice);

    bool* whether_walls_gpu;
    cudaMalloc((void**)&whether_walls_gpu, 2 * W * sizeof(bool));
    cudaMemcpy((void*)whether_walls_gpu, (void*)whether_walls, 2 * W * sizeof(bool), cudaMemcpyHostToDevice);

    bool* whether_floors_gpu;
    cudaMalloc((void**)&whether_floors_gpu, 2 * W * sizeof(bool));
    cudaMemcpy((void*)whether_floors_gpu, (void*)whether_floors, 2 * W * sizeof(bool), cudaMemcpyHostToDevice);

	float* ceilings_y_gpu;
	cudaMalloc((void**)&ceilings_y_gpu, 2 * W * sizeof(float));
    cudaMemcpy((void*)ceilings_y_gpu, (void*)ceilings_y, 2 * W * sizeof(float), cudaMemcpyHostToDevice);

	float* ceilings_x_gpu;
	cudaMalloc((void**)&ceilings_x_gpu, 2 * W * sizeof(float));
    cudaMemcpy((void*)ceilings_x_gpu, (void*)ceilings_x, 2 * W * sizeof(float), cudaMemcpyHostToDevice);

	float* floors_y_gpu;
	cudaMalloc((void**)&floors_y_gpu, 2 * W * sizeof(float));
    cudaMemcpy((void*)floors_y_gpu, (void*)floors_y, 2 * W * sizeof(float), cudaMemcpyHostToDevice);

	float* floors_x_gpu;
	cudaMalloc((void**)&floors_x_gpu, 2 * W * sizeof(float));
    cudaMemcpy((void*)floors_x_gpu, (void*)floors_x, 2 * W * sizeof(float), cudaMemcpyHostToDevice);


    //main function
    Annotate_gpu<<<(2 * W + 255)/ 256 ,256>>>(line_bottom_x_gpu, line_top_x_gpu, layout_seg_gpu, 
    whether_ceilings_gpu, whether_walls_gpu, whether_floors_gpu, ceilings_y_gpu, ceilings_x_gpu, floors_y_gpu, floors_x_gpu);
    cudaDeviceSynchronize();

    //move cuda data back to the main data
	cudaMemcpy((void*)whether_ceilings, (void*)whether_ceilings_gpu, 2 * W * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(whether_ceilings_gpu);

	cudaMemcpy((void*)whether_walls, (void*)whether_walls_gpu, 2 * W * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(whether_walls_gpu);

	cudaMemcpy((void*)whether_floors, (void*)whether_floors_gpu, 2 * W * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(whether_floors_gpu);

    cudaMemcpy((void*)ceilings_y, (void*)ceilings_y_gpu, 2 * W * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(ceilings_y_gpu);

    cudaMemcpy((void*)ceilings_x, (void*)ceilings_x_gpu, 2 * W * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(ceilings_x_gpu);

    cudaMemcpy((void*)floors_y, (void*)floors_y_gpu, 2 * W * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(floors_y_gpu);

    cudaMemcpy((void*)floors_x, (void*)floors_x_gpu, 2 * W * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(floors_x_gpu);
}
