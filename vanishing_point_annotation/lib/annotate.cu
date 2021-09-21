#include "annotate.hpp"
#include <iostream>
#include <stdio.h>


__constant__ int height; // the height of the picture
__constant__ int width; // the width of the picture
__constant__ int floor_label;  //the label of the floor in the layout segmentation
__constant__ int ceiling_label; //the label of the ceiling in the layout segmentation
__constant__ float vanishing_y, vanishing_x; //the place of the vanishing point
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__host__ __device__ static
double calculateSignedArea2(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c) {
	return ((c.x - a.x) * (b.y - a.y) - (b.x - a.x) * (c.y - a.y));
}








__global__ void Render_gpu(glm::vec3* positions, glm::ivec3* indices, int* color, int* findices, int* zbuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_primitives)
		return;

	glm::ivec3 face = indices[idx];
	glm::dvec3 p1 = glm::dvec3(rotation * positions[face[0]] + translation);
	glm::dvec3 p2 = glm::dvec3(rotation * positions[face[1]] + translation);
	glm::dvec3 p3 = glm::dvec3(rotation * positions[face[2]] + translation);

	glm::dvec3 p1_original = p1;
	glm::dvec3 p2_original = p2;
	glm::dvec3 p3_original = p3;


	//如果三个端点深度都是负数，那么一定不可见--剪枝。之前或逻辑并不对
	/*if (p1.z < 0.0001 || p2.z < 0.0001 || p3.z < 0.0001)
	{
		return;
	}*/
	//改变了剪枝逻辑
	if (p1.z < 0.0001 && p2.z < 0.0001 && p3.z < 0.0001)
	{
		return;
	}

	p1.z = 1.0f / p1.z;
	p2.z = 1.0f / p2.z;
	p3.z = 1.0f / p3.z;

	p1.x = p1.x * p1.z;
	p1.y = p1.y * p1.z;
	p2.x = p2.x * p2.z;
	p2.y = p2.y * p2.z;
	p3.x = p3.x * p3.z;
	p3.y = p3.y * p3.z;

	//这段代码不对：虽然X，Y是凸的，但是这里的值实际上是X/Z，Y/Z不是凸的，因此这样找min max不对（当然稠密mesh这样近似是对的）
	/*int minX = (MIN(p1.x, MIN(p2.x, p3.x)) * fx + cx);
	int minY = (MIN(p1.y, MIN(p2.y, p3.y)) * fy + cy);
	int maxX = (MAX(p1.x, MAX(p2.x, p3.x)) * fx + cx) + 0.999999f;
	int maxY = (MAX(p1.y, MAX(p2.y, p3.y)) * fy + cy) + 0.999999f;

	minX = MAX(0, minX);
	minY = MAX(0, minY);
	maxX = MIN(width, maxX);
	maxY = MIN(height, maxY);
	*/

	//改进了最小-最大值的求法：如果三个z都是正的就正常来，否则就遍历全局即可
	
	int minX = 0;
	int minY = 0;
	int maxX = width; 
	int maxY = height;
	if(p1.z > 0.0001 && p2.z > 0.0001 && p3.z > 0.0001)
	{
		minX = (MIN(p1.x, MIN(p2.x, p3.x)) * fx + cx);
		minY = (MIN(p1.y, MIN(p2.y, p3.y)) * fy + cy);
		maxX = (MAX(p1.x, MAX(p2.x, p3.x)) * fx + cx) + 0.999999f;
		maxY = (MAX(p1.y, MAX(p2.y, p3.y)) * fy + cy) + 0.999999f;
	
		minX = MAX(0, minX);
		minY = MAX(0, minY);
		maxX = MIN(width, maxX);
		maxY = MIN(height, maxY);
	}





	
	for (int py = minY; py <= maxY; ++py) {
		for (int px = minX; px <= maxX; ++px) {
			if (px < 0 || px >= width || py < 0 || py >= height)
				continue;

			float x = (px - cx) / fx;
			float y = (py - cy) / fy;

			glm::dvec3 baryCentricCoordinate = calculateBarycentricCoordinate(p1, p2, p3, glm::dvec3(x, y, 0));

			//这里也不对：X/Z Y/Z不在范围内不等于X,Y,Z实际不在范围内，我选择求出来然后反推
			//if (isBarycentricCoordInBounds(baryCentricCoordinate)) {
			int pixel = py * width + px;

			float inv_z = getZAtCoordinate(baryCentricCoordinate, p1, p2, p3);
			if(inv_z <= 0)
			{
				continue;
			}
			float real_x = x / inv_z;
			float real_y = y / inv_z;
			glm::dvec3 real_coordinate = calculateBarycentricCoordinate(p1_original, p2_original, p3_original, glm::dvec3(real_x, real_y, 0));
			if(isBarycentricCoordInBounds(real_coordinate)) {

				int z_quantize = inv_z * 100000;

				int original_z = atomicMax(&zbuffer[pixel], z_quantize);

				if (original_z < z_quantize) {
					glm::vec3 rgb = baryCentricCoordinate;
					if (render_primitives == 0) {
						atomicExchRGBZ(&zbuffer[pixel], &color[pixel], z_quantize, CompactRGBToInt(rgb));
					} else {
						atomicExchRGBZ(&zbuffer[pixel], &findices[pixel], z_quantize, idx);
					}
				}
			}
		}
	}
}

void Annotate(int H, int W, int floor_id, int ceiling_id, float vy, float vx, float* line_bottom_x, float* line_top_x, 
bool* whether_ceiling, bool* whether_wall, bool* whether_floor, float* ceiling_y, float* floor_y)


void Render(VertexBuffer& vertexBuffer, FrameBuffer& frameBuffer, int renderPrimitive) {
	cudaMemcpyToSymbol(height, &frameBuffer.row, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(width, &frameBuffer.col, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cx, &frameBuffer.cx, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cy, &frameBuffer.cy, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(fx, &frameBuffer.fx, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(fy, &frameBuffer.fy, sizeof(float), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(render_primitives, &renderPrimitive, sizeof(int), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(rotation, &vertexBuffer.rotation, sizeof(float) * 9, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(translation, &vertexBuffer.translation, sizeof(float) * 3, 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(num_primitives, &vertexBuffer.num_indices, sizeof(int), 0, cudaMemcpyHostToDevice);

	Render_gpu<<<(vertexBuffer.num_indices + 255) / 256, 256>>>(vertexBuffer.d_positions, vertexBuffer.d_indices, frameBuffer.d_colors, frameBuffer.d_findices, frameBuffer.d_z);
}
