from ctypes.wintypes import PINT
from operator import length_hint
import numpy as np
import cv2 as cv

def convert_img_to_array(coordinates, img):
    # extrear la informacion como un array 
    with open(coordinates, 'r') as t:
        datos = ''.join(t.readlines()).replace('\n', ';')

    m = np.matrix(datos)
    m = np.delete(m , 0, axis=1)
    print(m)

    # Extraer las dimensiones de la imagen a usar
    image = cv.imread(img)
    y, x = image.shape[:2]
    print('El alto es Y: ' + str(y) + ' y el ancho es X: ' + str(x))

    # Tranformacion de pixel para ubicar el punto medio de los BOXES
    i = 1
    for i in range(len(m)):
        print('Ancho en escala 0 - 1 en \n Y: ' + str(m[i , 1]) + ' \n X:' + str(m[i, 2]))
        transform_ruler3_y = y * m[i, 1]
        transform_ruler3_x = x * m[i, 2]
        print('Tamaño del Y: ' + str(transform_ruler3_y) + ' PX \n Tamaño del ancho X: ' + str(transform_ruler3_x) + " PX \n")
        i += 1


if __name__ == "__main__":
    convert_img_to_array('information.txt', '2f05e196-c6b2-4694-953c-daa69846cb1d.jpg')
