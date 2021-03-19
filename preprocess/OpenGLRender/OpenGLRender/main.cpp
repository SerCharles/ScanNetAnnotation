#include <freeglut/glut.h>
#include <iostream>
#include <math.h>
#include <windows.h>
#include <time.h>
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
float GlobalRefractionRate = 1;

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
