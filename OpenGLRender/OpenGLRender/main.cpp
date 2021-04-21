#include <freeglut/glut.h>
#include <iostream>
#include <math.h>
#include <windows.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "Geometry.hpp"
#include "Camera.hpp"
#include "MeshModel.hpp"
using namespace std;

#define OPENGL_LIGHT 1
#define MY_LIGHT 2
#define RAY_TRACE 3
#define RAY_TRACE_ACCELERATE 4

//全局常量
const int WindowSizeX = 640, WindowSizeY = 480, WindowPlaceX = 100, WindowPlaceY = 100;
const char WindowName[] = "MyScene";
string BaseDir = "E:\\dataset\\scannet\\scans\\scene0000_00\\annotation\\pictures";

void SavePicture(int id)
{
	string save_place = BaseDir + "\\" + to_string(id) + ".jpg";
	//save image
	GLubyte* pPixelData;
	pPixelData = (GLubyte*)malloc(WindowSizeX * WindowSizeY * 4);//分配内存
	if (pPixelData == 0) return;
	glReadBuffer(GL_FRONT);//保存窗口渲染的结果
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);//解压窗口数据结构
	glReadPixels(0, 0, WindowSizeX, WindowSizeY, GL_RGBA, GL_UNSIGNED_BYTE, pPixelData);//存储像素数据

	cv::Mat img;
	std::vector<cv::Mat> imgPlanes;
	img.create(WindowSizeY, WindowSizeX, CV_8UC3);//确定图片通道和尺寸
	cv::split(img, imgPlanes);//将图像按照通道数拆分，三个单通道序列

	for (int i = 0; i < WindowSizeY; i++) {
		unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);//B
		unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);//G
		unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);//R
		//opencv里面以BGR存储的，而Mac上opengl是RGBA，所以需要改变顺序保存
		for (int j = 0; j < WindowSizeX; j++) {
			int k = 4 * (i * WindowSizeX + j);//RGBA数据结构，不需要A，跳过，所以步长乘以4
			plane2Ptr[j] = pPixelData[k];//R
			plane1Ptr[j] = pPixelData[k + 1];//G
			plane0Ptr[j] = pPixelData[k + 2];//B
		}
	}
	cv::merge(imgPlanes, img);//合并多通道图像
	cv::flip(img, img, 0); // 反转图像，因为opengl和opencv的坐标系y轴是相反的
	//cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);//转换为灰度图
	//cv::namedWindow("openglGrab");
	//cv::imshow("openglGrab", img);
	//cv::waitKey();
	cv::imwrite(save_place, img);//保存图片

}


//光照，相机
Camera TheCamera;

//面片物体
MeshModel Scene;


//初始化函数集合
//初始化窗口
void InitWindow()
{
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WindowSizeX, WindowSizeY);
	glutInitWindowPosition(WindowPlaceX, WindowPlaceY);
	glutCreateWindow(WindowName);
	const GLubyte* OpenGLVersion = glGetString(GL_VERSION);
	const GLubyte* gluVersion = gluGetString(GLU_VERSION);
	printf("OpenGL实现的版本号：%s\n", OpenGLVersion);
	printf("OGLU工具库版本：%s\n", gluVersion);
}


//初始化相机
void InitCamera()
{
	//设置初始相机位置
	string filename = "poses.txt";
	TheCamera.Init(filename);
}



//初始化的主函数
void InitScene()
{
	InitCamera();
	glEnable(GL_DEPTH_TEST);
	Scene.InitPlace("planes_with_line.ply");
}

//绘制函数集合
//设置相机位置
void SetCamera()
{
	glLoadIdentity();
	Point eye = TheCamera.Eye;
	Point center = TheCamera.Center;
	Point up = TheCamera.Up;

	gluLookAt(eye.x, eye.y, eye.z, center.x, center.y, center.z, up.x, up.y, up.z);
}





//绘制的主函数
void DrawScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//清除颜色缓存
	SetCamera();//设置相机
	
	
	for (int i = 0; i < Scene.FaceNum; i++)
	{
		Mesh the_face = Scene.Faces[i];
		DrawMesh(the_face);
	}
	
	for (int i = 0; i < Scene.LineNum; i++)
	{
		Line the_line = Scene.Lines[i];
		DrawLine(the_line);
	}
	glFlush();	
	glutSwapBuffers();
	SavePicture(TheCamera.IDList[TheCamera.CurrentNum]);
	TheCamera.CurrentNum = (TheCamera.CurrentNum + 1) % TheCamera.PoseNum;
	TheCamera.ResetCurrentPlace();
}

//全局定时器
void OnTimer(int value)
{
	glutPostRedisplay();//标记当前窗口需要重新绘制，调用myDisplay()
	glutTimerFunc(20, OnTimer, 1);
}



//处理键盘点击（WASD）
void OnKeyClick(unsigned char key, int x, int y)
{
	int type = -1;
	if (key == 'w')
	{
		type = 0;
	}
	else if (key == 'a')
	{
		type = 1;
	}
	else if (key == 's')
	{
		type = 2;
	}
	else if (key == 'd')
	{
		type = 3;
	}
	TheCamera.KeyboardMove(type);
}

//处理键盘点击（前后左右）
void OnSpecialKeyClick(GLint key, GLint x, GLint y)
{
	int type = -1;
	if (key == GLUT_KEY_UP)
	{
		type = 0;
	}
	if (key == GLUT_KEY_LEFT)
	{
		type = 1;
	}
	if (key == GLUT_KEY_DOWN)
	{
		type = 2;
	}
	if (key == GLUT_KEY_RIGHT)
	{
		type = 3;
	}
	TheCamera.KeyboardMove(type);
}

//reshape函数
void Reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(75.0f, (float)w / h, 1.0f, 1000.0f);
	glMatrixMode(GL_MODELVIEW);
}





int main(int argc, char**argv)
{
	glutInit(&argc, argv);
	InitWindow();             //初始化窗口
	InitScene();              //初始化场景
	glutReshapeFunc(Reshape); //绑定reshape函数
	glutDisplayFunc(DrawScene); //绑定显示函数
	glutTimerFunc(20, OnTimer, 1);  //启动计时器
	glutKeyboardFunc(OnKeyClick);//绑定键盘点击函数
	glutSpecialFunc(OnSpecialKeyClick);//绑定特殊键盘点击函数
	glutMainLoop();

	return 0;
}
