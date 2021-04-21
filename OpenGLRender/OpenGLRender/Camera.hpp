/*
描述：OpenGL相机视角定义
日期：2020/11/6
*/
#pragma once
#include<math.h>
#include<fstream>
#include"Geometry.hpp"
using namespace std;


class Camera
{
public:
	vector<Point> EyeList;
	vector<Point> CenterList;
	vector<Point> UpList;
	vector<int> IDList;
	int PoseNum;
	int CurrentNum;
	int ID;
	Point Eye;
	Point Center;
	Point Up;

public:
	Camera()
	{
		EyeList.clear();
		CenterList.clear();
		UpList.clear();
		IDList.clear();
	}

	void Init(string filename)
	{
		fstream f(filename);
		f >> PoseNum;
		for (int i = 0; i < PoseNum; i++)
		{
			int id;
			f >> id;
			IDList.push_back(id);

			float x, y, z;
			f >> x >> y >> z;
			Point eye(x, y, z);
			EyeList.push_back(eye);

			f >> x >> y >> z;
			Point center(x, y, z);
			CenterList.push_back(center);

			f >> x >> y >> z;
			Point up(x, y, z);
			UpList.push_back(up);
		}
		CurrentNum = 0;
		ID = IDList[CurrentNum];
		Eye = EyeList[CurrentNum];
		Center = CenterList[CurrentNum];
		Up = UpList[CurrentNum];
		f.close();
	}



	void ResetCurrentPlace()
	{
		ID = IDList[CurrentNum];
		Eye = EyeList[CurrentNum];
		Center = CenterList[CurrentNum];
		Up = UpList[CurrentNum];
	}


	//处理键盘移动事件，更改水平位置和视点中心
	void KeyboardMove(int type)
	{
		float change_x = 0;
		float change_z = 0;
		
		//0123代表WASD
		if (type == 0)
		{
			CurrentNum -= 1;
			if (CurrentNum < 0)
			{
				CurrentNum += PoseNum;
			}
		}
		else if (type == 1)
		{
			CurrentNum -= 1;
			if (CurrentNum < 0)
			{
				CurrentNum += PoseNum;
			}
		}
		else if (type == 2)
		{
			CurrentNum = (CurrentNum + 1) % PoseNum;
		}
		else if (type == 3)
		{
			CurrentNum = (CurrentNum + 1) % PoseNum;
		}

		ResetCurrentPlace();
	}
	
};