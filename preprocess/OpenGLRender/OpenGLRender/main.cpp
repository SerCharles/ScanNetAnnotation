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

//ȫ�ֳ���
const int WindowSizeX = 640, WindowSizeY = 480, WindowPlaceX = 100, WindowPlaceY = 100;
const char WindowName[] = "MyScene";
float GlobalRefractionRate = 1;

//���գ����
Camera TheCamera;

//��Ƭ����
MeshModel Scene;


//��ʼ����������
//��ʼ������
void InitWindow()
{
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WindowSizeX, WindowSizeY);
	glutInitWindowPosition(WindowPlaceX, WindowPlaceY);
	glutCreateWindow(WindowName);
	const GLubyte* OpenGLVersion = glGetString(GL_VERSION);
	const GLubyte* gluVersion = gluGetString(GLU_VERSION);
	printf("OpenGLʵ�ֵİ汾�ţ�%s\n", OpenGLVersion);
	printf("OGLU���߿�汾��%s\n", gluVersion);
}


//��ʼ�����
void InitCamera()
{
	//���ó�ʼ���λ��
	string filename = "poses.txt";
	TheCamera.Init(filename);
}



//��ʼ����������
void InitScene()
{
	InitCamera();
	glEnable(GL_DEPTH_TEST);
	Scene.InitPlace("planes_with_line.ply");
}

//���ƺ�������
//�������λ��
void SetCamera()
{
	glLoadIdentity();
	Point eye = TheCamera.Eye;
	Point center = TheCamera.Center;
	Point up = TheCamera.Up;

	gluLookAt(eye.x, eye.y, eye.z, center.x, center.y, center.z, up.x, up.y, up.z);
}





//���Ƶ�������
void DrawScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//�����ɫ����
	SetCamera();//�������
	
	
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

//ȫ�ֶ�ʱ��
void OnTimer(int value)
{
	glutPostRedisplay();//��ǵ�ǰ������Ҫ���»��ƣ�����myDisplay()
	glutTimerFunc(20, OnTimer, 1);
}



//������̵����WASD��
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

//������̵����ǰ�����ң�
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

//reshape����
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
	InitWindow();             //��ʼ������
	InitScene();              //��ʼ������
	glutReshapeFunc(Reshape); //��reshape����
	glutDisplayFunc(DrawScene); //����ʾ����
	glutTimerFunc(20, OnTimer, 1);  //������ʱ��
	glutKeyboardFunc(OnKeyClick);//�󶨼��̵������
	glutSpecialFunc(OnSpecialKeyClick);//��������̵������
	glutMainLoop();

	return 0;
}
