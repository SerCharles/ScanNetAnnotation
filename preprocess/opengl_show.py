import os
import json
import numpy as np
from plyfile import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class GLShow:
    def drawOneEdge(self, point_a, point_b):
        '''
        description: draw one line with opengl
        parameter: two points
        return: empty
        '''
        x_a, y_a, z_a, r_a, g_a, b_a = point_a
        x_b, y_b, z_b, r_b, g_b, b_b = point_b

        x_a = float(x_a)
        y_a = float(y_a)
        z_a = float(z_a)
        r_a = float(r_a / 255)
        g_a = float(g_a / 255)
        b_a = float(b_a / 255)
        x_b = float(x_b)
        y_b = float(y_b)
        z_b = float(z_b)
        r_b = float(r_b / 255)
        g_b = float(g_b / 255)
        b_b = float(b_b / 255)


        glBegin(GL_LINES)
        glColor3f(r_a, g_a, b_a)     
        glVertex3f(x_a, y_a, z_a)   
        glColor3f(r_b, g_b, b_b)      
        glVertex3f(x_b, y_b, z_b)         
        glEnd()
        glFlush()


    def drawOneFace(self, point_a, point_b, point_c):
        '''
        description: draw one triangle mesh with opengl
        parameter: three points
        return: empty
        '''
        x_a, y_a, z_a, r_a, g_a, b_a = point_a
        x_b, y_b, z_b, r_b, g_b, b_b = point_b
        x_c, y_c, z_c, r_c, g_c, b_c = point_c

        x_a = float(x_a)
        y_a = float(y_a)
        z_a = float(z_a)
        r_a = float(r_a / 255)
        g_a = float(g_a / 255)
        b_a = float(b_a / 255)
        x_b = float(x_b)
        y_b = float(y_b)
        z_b = float(z_b)
        r_b = float(r_b / 255)
        g_b = float(g_b / 255)
        b_b = float(b_b / 255)
        x_c = float(x_c)
        y_c = float(y_c)
        z_c = float(z_c)
        r_c = float(r_c / 255)
        g_c = float(g_c / 255)
        b_c = float(b_c / 255)

        glBegin(GL_POLYGON)
        glColor3f(r_a, g_a, b_a)     
        glVertex3f(x_a, y_a, z_a)   
        glColor3f(r_b, g_b, b_b)      
        glVertex3f(x_b, y_b, z_b)       
        glColor3f(r_c, g_c, b_c)       
        glVertex3f(x_c, y_c, z_c)    
        glEnd()
        glFlush()
    

    def drawScene(self):
        '''
        description: draw all triangle meshs and lines with opengl
        parameter: empty
        return: empty
        '''
        glClear(GL_COLOR_BUFFER_BIT)

        for face in self.faces: 
            a, b, c = face[0]
            point_a = self.vertexs[a]
            point_b = self.vertexs[b]
            point_c = self.vertexs[c]
            self.drawOneFace(point_a, point_b, point_c)

        for edge in self.edges: 
            a, b, r, g, bb = edge
            point_a = self.vertexs[a]
            point_b = self.vertexs[b]
            self.drawOneEdge(point_a, point_b)

        glFlush()

    def __init__(self, ROOT_FOLDER, scene_id):
        self.id = scene_id
        ply_name = os.path.join(ROOT_FOLDER, scene_id, 'annotation', 'planes_with_line.ply')
        plydata = PlyData.read(ply_name)
        self.vertexs = plydata['vertex']
        self.faces = plydata['face']
        try:
            self.edges = plydata['edge']
        except: 
            self.edges = []

        glutInit()
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
        glutInitWindowPosition(200, 200)
        glutInitWindowSize(800, 800)
        glutCreateWindow("Show " + scene_id)
        #glutReshapeFunc(Reshape)
        glutDisplayFunc(self.drawScene)
        #glutIdleFunc(self.drawScene)

        glutMainLoop()

GLShow('E:\\dataset\\scannet\\scans', 'scene0000_00')
