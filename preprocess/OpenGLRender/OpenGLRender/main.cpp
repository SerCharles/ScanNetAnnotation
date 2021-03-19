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

//ȫ�ֳ���
const int WindowSizeX = 640, WindowSizeY = 480, WindowPlaceX = 100, WindowPlaceY = 100;
const char WindowName[] = "MyScene";
string BaseDir = "E:\\dataset\\scannet\\scans\\scene0000_00\\annotation\\pictures";

void SavePicture(int id)
{
	string save_place = BaseDir + "\\" + to_string(id) + ".jpg";
	//save image
	GLubyte* pPixelData;
	pPixelData = (GLubyte*)malloc(WindowSizeX * WindowSizeY * 4);//�����ڴ�
	if (pPixelData == 0) return;
	glReadBuffer(GL_FRONT);//���洰����Ⱦ�Ľ��
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);//��ѹ�������ݽṹ
	glReadPixels(0, 0, WindowSizeX, WindowSizeY, GL_RGBA, GL_UNSIGNED_BYTE, pPixelData);//�洢��������

	cv::Mat img;
	std::vector<cv::Mat> imgPlanes;
	img.create(WindowSizeY, WindowSizeX, CV_8UC3);//ȷ��ͼƬͨ���ͳߴ�
	cv::split(img, imgPlanes);//��ͼ����ͨ������֣�������ͨ������

	for (int i = 0; i < WindowSizeY; i++) {
		unsigned char* plane0Ptr = imgPlanes[0].ptr<unsigned char>(i);//B
		unsigned char* plane1Ptr = imgPlanes[1].ptr<unsigned char>(i);//G
		unsigned char* plane2Ptr = imgPlanes[2].ptr<unsigned char>(i);//R
		//opencv������BGR�洢�ģ���Mac��opengl��RGBA��������Ҫ�ı�˳�򱣴�
		for (int j = 0; j < WindowSizeX; j++) {
			int k = 4 * (i * WindowSizeX + j);//RGBA���ݽṹ������ҪA�����������Բ�������4
			plane2Ptr[j] = pPixelData[k];//R
			plane1Ptr[j] = pPixelData[k + 1];//G
			plane0Ptr[j] = pPixelData[k + 2];//B
		}
	}
	cv::merge(imgPlanes, img);//�ϲ���ͨ��ͼ��
	cv::flip(img, img, 0); // ��תͼ����Ϊopengl��opencv������ϵy�����෴��
	//cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);//ת��Ϊ�Ҷ�ͼ
	//cv::namedWindow("openglGrab");
	//cv::imshow("openglGrab", img);
	//cv::waitKey();
	cv::imwrite(save_place, img);//����ͼƬ

}


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
	SavePicture(TheCamera.IDList[TheCamera.CurrentNum]);
	TheCamera.CurrentNum = (TheCamera.CurrentNum + 1) % TheCamera.PoseNum;
	TheCamera.ResetCurrentPlace();
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
