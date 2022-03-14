from ctypes.wintypes import PINT
from operator import length_hint
import numpy as np
import cv2 as cv

def convert_img_to_array(coordinates, img):
    # extrear la informacion como un array 
    with open(coordinates, 'r') as t:
        datos = ''.join(t.readlines()).replace('\n', ';')

    m = np.matrix(datos)
    print(m)

    # Extraer las dimensiones de la imagen a usar
    image = cv.imread(img)
    y, x = image.shape[:2]
    print('El alto es Y: ' + str(y) + ' y el ancho es X: ' + str(x))

    # Tranformacion de pixel para ingresar los datos en el OCR
    i = 1
    for i in range(len(m)):
        print('Fila de contador numero ' + str(i))
        print('Ancho en escala 0 - 1 en \n alto: ' + str(m[i , 1]) + ' \nancho:' + str(m[i, 2]) + '\nTercer columna: ' + str(m[i, 3]) + '\nCuarta columna: ' + str(m[i, 4]) + '\n')
        transform_ruler3_y = y * m[i, 1]
        transform_ruler3_x = x * m[i, 2]
        transform_ruler3_3 = y * m[i, 3]
        transform_ruler3_4 = x * m[i, 4]

        print('Tamaño del alto en Y: ' + str(transform_ruler3_y) + ' PX \nTamaño del ancho X: ' + str(transform_ruler3_x) + " PX \nInformacion en pixels de la tercera columna: " + str(transform_ruler3_3) + 'PX \nInformacion en pixels de la cuarta columna: ' + str(transform_ruler3_4) + 'PX \n')

        # OBTENER LA IMAGEN RECORTADA CON LAS COORDENADAS OBTENIDAS
        pixelsY = int(y * m[i, 1])
        pixelsX = int(x * m[i, 2])
        pixelsH = int(y * m[i, 3])
        pixelsW = int(x * m[i, 4])
        
        image_out = image[pixelsY:pixelsX, pixelsH:pixelsW]
        cv.imshow('Imagen recortada', image_out)
        cv.waitKey(0)
        i += 1


if __name__ == "__main__":
    convert_img_to_array('information.txt', '2f05e196-c6b2-4694-953c-daa69846cb1d.jpg')
