/*
描述：3D点类和各种常用函数
日期：2020/11/6
*/
#pragma once


#include<math.h>
#include<GL/glut.h>
#include<vector>
using namespace std;

#define PI 3.1415926535

//3D点的基类
class Point
{
public:
	float x;
	float y;
	float z;
	Point()
	{
		x = 0;
		y = 0;
		z = 0;
	}
	Point(float tx, float ty, float tz)
	{
		x = tx;
		y = ty;
		z = tz;
	}
	void SetPlace(float tx, float ty, float tz)
	{
		x = tx;
		y = ty;
		z = tz;
	}
	Point operator+(const Point& b)
	{
		Point c;
		c.x = x + b.x;
		c.y = y + b.y;
		c.z = z + b.z;
		return c;
	}
	Point operator-(const Point& b)
	{
		Point c;
		c.x = x - b.x;
		c.y = y - b.y;
		c.z = z - b.z;
		return c;
	}
	Point operator*(const float& b)
	{
		Point c;
		c.x = x * b;
		c.y = y * b;
		c.z = z * b;
		return c;
	}
	float operator*(const Point& b)
	{
		float sum = 0;
		sum += x * b.x;
		sum += y * b.y;
		sum += z * b.z;
		return sum;
	}
	Point operator/(const float& b)
	{
		Point c;
		c.x = x / b;
		c.y = y / b;
		c.z = z / b;
		return c;
	}


	float Square()
	{
		return x * x + y * y + z * z;
	}

	float Dist()
	{
		return sqrt(Square());
	}

	void Normalize()
	{
		float dist = Dist();
		x /= dist;
		y /= dist;
		z /= dist;
	}
};



class Color
{
public:
	float R;
	float G;
	float B;
	Color()
	{
		R = 0;
		G = 0;
		B = 0;
	}
	Color(float r, float g, float b)
	{
		R = r;
		G = g;
		B = b;
	}
	Color(int r, int g, int b)
	{
		R = float(r) / 255.0;
		G = float(g) / 255.0;
		B = float(b) / 255.0;
	}
	Color operator+(const Color& b)
	{
		Color c;
		c.R = R + b.R;
		c.G = G + b.G;
		c.B = B + b.B;
		return c;
	}
	Color operator-(const Color& b)
	{
		Color c;
		c.R = R - b.R;
		c.G = G - b.G;
		c.B = B - b.B;
		return c;
	}
	Color operator*(const float& b)
	{
		Color c;
		c.R = R * b;
		c.G = G * b;
		c.B = B * b;
		return c;
	}

	Color operator/(const float& b)
	{
		Color c;
		c.R = R / b;
		c.G = G / b;
		c.B = B / b;
		return c;
	}
};

class Vertex
{
public:
	Point place; 
	Color color;
	Vertex() {}
	Vertex(Point p, Color c)
	{
		place = p;
		color = c;
	}
};

class Line
{
public:
	Vertex start;
	Vertex end;
	Line() {}
	Line(Vertex s, Vertex t)
	{
		start = s;
		end = t;
	}
	Line(vector<Vertex>& vertexs, int s, int t)
	{
		start = vertexs[s];
		end = vertexs[t];
	}
};

class Mesh
{
public:
	Vertex a;
	Vertex b;
	Vertex c;
	Mesh() {}
	Mesh(Vertex aa, Vertex bb, Vertex cc)
	{
		a = aa;
		b = bb;
		c = cc;
	}
	Mesh(vector<Vertex>& vertexs, int aa, int bb, int cc)
	{
		a = vertexs[aa];
		b = vertexs[bb];
		c = vertexs[cc];
	}
};


//绘制一条线
void DrawLine(Line l)
{
	glBegin(GL_LINES);
	glColor3f(l.start.color.R, l.start.color.G, l.start.color.B);
	glVertex3f(l.start.place.x, l.start.place.y, l.start.place.z);
	glColor3f(l.end.color.R, l.end.color.G, l.end.color.B);
	glVertex3f(l.end.place.x, l.end.place.y, l.end.place.z);
	glEnd();
}

//绘制一个三角形
void DrawMesh(Mesh m)
{
	glBegin(GL_POLYGON);
	glColor3f(m.a.color.R, m.a.color.G, m.a.color.B);
	glVertex3f(m.a.place.x, m.a.place.y, m.a.place.z);
	glColor3f(m.b.color.R, m.b.color.G, m.b.color.B);
	glVertex3f(m.b.place.x, m.b.place.y, m.b.place.z);
	glColor3f(m.c.color.R, m.c.color.G, m.c.color.B);
	glVertex3f(m.c.place.x, m.c.place.y, m.c.place.z);
	glEnd();
}



/*
描述：判断点是否在多边形内，用角度法求点和每条边交点夹角和，为2pi才在内否则在外
参数：点，多边形
返回：是/否
*/
bool JudgeInside(Point& p, vector<Point>& polygon)
{
	float sum_arc = 0;
	for (int i = 0; i < polygon.size(); i++)
	{
		Point current = polygon[i];
		Point next;
		if (i == polygon.size() - 1)
		{
			next = polygon[0];
		}
		else
		{
			next = polygon[i + 1];
		}
		float a = (current - p).Dist();
		float b = (next - p).Dist();
		float c = (current - next).Dist();
		float up = a * a + b * b - c * c;
		float down = 2 * a * b;
		if (down == 0) return 0;
		float arc = acos(up / down);
		sum_arc += arc;
	}
	if (abs(sum_arc - 2 * PI) < 0.01) return 1;
	else return 0;
}
