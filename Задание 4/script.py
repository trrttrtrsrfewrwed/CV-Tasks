import pygame
import random
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import *

import numpy as np
import cv2

vertex_shader = """
varying vec3 vN;
varying vec3 v;
varying vec4 color;
void main(void)  
{     
   v = vec3(gl_ModelViewMatrix * gl_Vertex);       
   vN = normalize(gl_NormalMatrix * gl_Normal);
   color = gl_Color;
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;  
}
"""

fragment_shader = """
varying vec3 vN;
varying vec3 v; 
varying vec4 color;
#define MAX_LIGHTS 1 
void main (void) 
{ 
   vec3 N = normalize(vN);
   vec4 finalColor = vec4(0.0, 0.0, 0.0, 0.0);

   for (int i=0;i<MAX_LIGHTS;i++)
   {
      vec3 L = normalize(gl_LightSource[i].position.xyz - v); 
      vec3 E = normalize(-v); // we are in Eye Coordinates, so EyePos is (0,0,0) 
      vec3 R = normalize(-reflect(L,N)); 

      vec4 Iamb = gl_LightSource[i].ambient; 
      vec4 Idiff = gl_LightSource[i].diffuse * max(dot(N,L), 0.0);
      Idiff = clamp(Idiff, 0.0, 1.0); 
      vec4 Ispec = gl_LightSource[i].specular * pow(max(dot(R,E),0.0),0.3*gl_FrontMaterial.shininess);
      Ispec = clamp(Ispec, 0.0, 1.0); 

      finalColor += Iamb + Idiff + Ispec;
   }
   gl_FragColor = color * finalColor; 
}
"""

textureCoordinates = ((0, 0), (0, 1), (1, 1), (1, 0))

surfaces = (
    (0, 1, 2, 3),
    )

normals = [
    (0,  0, 1),  # surface 0
]


def draw_rect(verticies, normals, surfaces):
    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    for i_surface, surface in enumerate(surfaces):
        glNormal3fv(normals[i_surface])
        for i_vertex, vertex in enumerate(surface):
            glTexCoord2fv(textureCoordinates[i_vertex])
            glVertex3fv(verticies[vertex])
    glEnd()


def add_texture(name):
    image = pygame.image.load(name)
    datas = pygame.image.tostring(image, 'RGBA')
    texID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texID)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.get_width(), image.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, datas)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    glEnable(GL_TEXTURE_2D)


def lamp_lightning(lamp_x=0.5, lamp_y=0.5):
    glLight(GL_LIGHT0, GL_POSITION, (lamp_x, lamp_y, 1, 1))
    # Diffuse lighting
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    return


def lightning():
    glLight(GL_LIGHT0, GL_POSITION,  (0, 0, 3, 1))
    # Diffuse lighting
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (255/255, 225/255, 195/255, 1))
    return


def process(n_iter,
            img_name='1.png',
            rotate_angle_x=5,
            rotate_angle_y=5,
            rotate_angle_z=5,
            lamp_x=0.5,
            lamp_y=0.5):
    table_img = cv2.imread('table.png')
    table_w, table_h = table_img.shape[0:2]
    table_verticies = (
        (-4, -4 * table_w / table_h, -0.01),  # 0
        (-4, 4 * table_w / table_h, -0.01),  # 1
        (4, 4 * table_w / table_h, -0.01),  # 2
        (4, -4 * table_w / table_h, -0.01),  # 3
    )

    img = cv2.imread(img_name)
    w, h = img.shape[0:2]

    verticies = (
        (-1, -w / h, 0),  # 0
        (-1, w / h, 0),  # 1
        (1, w / h, 0),  # 2
        (1, -w / h, 0),  # 3
    )

    pygame.init()
    display = (4000, 4000)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # shaders for "lamp" light
    program = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    # Setting camera position and orientation
    gluLookAt(0.0, 0.0, 0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)

    # Setting camera's field of view
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glMatrixMode(GL_MODELVIEW)

    glTranslatef(0, 0, -5)

    glRotatef(rotate_angle_z, 0, 0, 1)
    glRotatef(rotate_angle_x, 1, 0, 0)
    glRotatef(rotate_angle_y, 0, 1, 0)

    # "lamp" light
    lamp_lightning(lamp_x, lamp_y)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glUseProgram(program)
    # Specular Lighting
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 5)
    draw_rect(table_verticies, normals, surfaces)

    # Specular Lighting
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 128)
    draw_rect(verticies, normals, surfaces)

    glDisable(GL_LIGHT0)
    glDisable(GL_LIGHTING)
    glDisable(GL_COLOR_MATERIAL)

    image_buffer = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
    image1 = np.frombuffer(image_buffer, dtype=np.uint8).reshape(display[0], display[1], 3)
    cv2.imwrite("image1.png", cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))

    # basic light
    lightning()

    glUseProgram(0)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    add_texture("table.png")
    draw_rect(table_verticies, normals, surfaces)

    add_texture(img_name)
    draw_rect(verticies, normals, surfaces)

    glDisable(GL_LIGHT0)
    glDisable(GL_LIGHTING)
    glDisable(GL_COLOR_MATERIAL)

    image_buffer = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
    image2 = np.frombuffer(image_buffer, dtype=np.uint8).reshape(display[0], display[1], 3)
    cv2.imwrite("image2.png", cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))

    grayscale = np.dot(image1, [0.114, 0.587, 0.299])
    res = cv2.cvtColor(np.array(image2 * np.repeat(np.expand_dims(grayscale, axis=-1), 3, axis=-1) / 255
                                                    * 0.9 + image1 * 0.1, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("image3.png", res)

    # with Image.from_array(res) as img:
    #     # Generate noise image using spread() function
    #     img.noise("poisson", attenuate=0.9)
    #     img.save(filename="image4.png")


    # # Generate Gaussian noise
    # gauss = np.random.normal(0, 1, res.size)
    # gauss = gauss.reshape(res.shape).astype('uint8')
    # # Add the Gaussian noise to the image
    # img_gauss = cv2.add(res, gauss)
    #
    # cv2.imwrite("image4.png", img_gauss)

    # cv2.imwrite(img_name[:-4] + "_" + str(n_iter) + "_out.png", res)
    pygame.quit()


def main():
    rotate_angle_x = random.randrange(-6, 6)
    rotate_angle_y = random.randrange(-6, 6)
    rotate_angle_z = random.randrange(-6, 6)
    lamp_x = random.uniform(-0.5, 0.5)
    lamp_y = random.uniform(-0.5, 0.5)
    process(0, "1.png", rotate_angle_x, rotate_angle_y, rotate_angle_z, lamp_x, lamp_y)

    # N_IMAGES = 10
    # N_ITERS = 10
    # for i in range(N_IMAGES):
    #     img_name = str(i + 1) + ".png"
    #     for j in range(N_ITERS):
    #         rotate_angle_x = random.randrange(-6, 6)
    #         rotate_angle_y = random.randrange(-6, 6)
    #         rotate_angle_z = random.randrange(-6, 6)
    #         lamp_x = random.uniform(-0.5, 0.5)
    #         lamp_y = random.uniform(-0.5, 0.5)
    #         process(j, img_name, rotate_angle_x, rotate_angle_y, rotate_angle_z, lamp_x, lamp_y)


main()