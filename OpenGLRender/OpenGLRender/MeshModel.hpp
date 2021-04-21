/*
描述：三角面片和模型类
日期：2020/11/6
*/
#pragma once
#include"Geometry.hpp"
#include<iostream>
#include<cstdlib>
#include<fstream>
#include<freeglut/glut.h>
#include<string>
#include<vector>
#include<cmath>
using namespace std;



class MeshModel
{
public:
	//存储位置
	string BaseDir = "";
	string FullDir = BaseDir;
	//位置信息：大小和中心；点和面片的个数；点和面片的位置以及面片的点id
	int VertexNum;
	int FaceNum;
	int LineNum;
	vector<Vertex> Vertexs;
	vector<Mesh> Faces;
	vector<Line> Lines;
	MeshModel() {}



	/*
	描述：string转int
	参数：string
	返回：int
	*/
	int StringToInt(string str)
	{
		int length = str.size();
		int int_base = 1;
		int sum = 0;
		for (int i = length - 1; i >= 0; i--)
		{
			sum += (str[i] - '0') * int_base;
			int_base *= 10;
		}
		return sum;
	}

	/*
	描述：读取ply文件头
	参数：文件流
	返回：空
	*/
	void ReadPLYHead(fstream& f)
	{
		int line_num = 0;
		while (1)
		{

			string line;
			string end_line = "end_header";
			getline(f, line);
			if (line_num == 2)
			{
				string vertex_num = line.substr(15);
				VertexNum = StringToInt(vertex_num);
			}
			else if (line_num == 9)
			{
				string face_num = line.substr(13);
				FaceNum = StringToInt(face_num);
			}
			else if (line_num == 11)
			{
				string line_num = line.substr(13);
				LineNum = StringToInt(line_num);
			}
			else if (line == end_line)
			{
				break;
			}
			line_num++;

		}
	}

	/*
	描述：初始化面片
	参数：文件名
	返回：无
	*/
	void InitPlace(string filename)
	{
		FullDir += filename;
		VertexNum = 0;
		FaceNum = 0;
		fstream f(FullDir);

		ReadPLYHead(f);

		for (int i = 0; i < VertexNum; i++)
		{
			float x, y, z;
			int r, g, b;
			f >> x >> y >> z >> r >> g >> b;
			Point nova_place(x, y, z);
			Color nova_color(r, g, b);
			Vertex nova(nova_place, nova_color);
			Vertexs.push_back(nova);
		}


		for (int i = 0; i < FaceNum; i++)
		{
			int num, v1_num, v2_num, v3_num;
			f >> num >> v1_num >> v2_num >> v3_num;
			Mesh nova(Vertexs, v1_num, v2_num, v3_num);
			Faces.push_back(nova);
		}

		for (int i = 0; i < LineNum; i++)
		{
			int s_num, t_num;
			f >> s_num >> t_num;
			Line nova(Vertexs, s_num, t_num);
			Lines.push_back(nova);
		}
	}



};


