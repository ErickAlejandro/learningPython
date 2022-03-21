import numpy as np
import cv2
from math import ceil
import pytesseract as pyt


def ocr(bbox):
    # Lectura de los datos con Pytesseract
    text = pyt.image_to_string(bbox)
    text = text.strip('\n\x0c')
    text = text.replace('\n', ' ')
    text = text.replace(' -', '-')
    text = text.replace('- ', '')
    text = text.replace('&L ', '')
    print('Informacion de la fila: ' + str(text))
    return text
    #plt.show()


def convert_img_to_array(coordinates, img):
    # extrear la informacion como un array
    with open(coordinates, 'r') as t:
        datos = ''.join(t.readlines()).replace('\n', ';')

    m = np.matrix(datos)
    information = []

    # ordenar la matriz
    m = np.array(m)
    m = m[m[:, 0].argsort()]
    print(m)

    # Extraer las dimensiones de la imagen a usar
    image = cv2.imread(img)
    y, x = image.shape[:2]
    print('\nEl ancho: ' + str(x) + ' y el alto: ' + str(y) + '\n')

    # Tranformacion de pixel para ingresar los datos en el OCR
    i = 1
    for i in range(len(m)):
        print('Fila de contador numero ' + str(i))
        print('Ancho en escala 0 - 1 en \nPunto medio en X: ' + str(m[i, 1]) + '\nPunto medio en Y:' + str(
            m[i, 2]) + '\nAncho: ' + str(m[i, 3]) + '\nAlto: ' + str(m[i, 4]) + '\n')
        x_center = x * m[i, 1]
        y_center = y * m[i, 2]
        transform_ruler3_w = (x * m[i, 3]) / 2
        transform_ruler3_h = (y * m[i, 4]) / 2

        print('Escala en PIXELS \nPunto medio en X: ' + str(x_center) + ' PX \nPunto medio en Y: ' + str(y_center) +
              " PX \nAncho: " + str(transform_ruler3_w) + ' PX \nAlto: ' + str(transform_ruler3_h) + ' PX \n')

        # Reconstruccion del bondibox (Informacion del bondybox)
        x0 = ceil(x_center - transform_ruler3_w)
        y0 = ceil(y_center - transform_ruler3_h)

        x1 = ceil(x_center + transform_ruler3_w)
        y1 = ceil(y_center + transform_ruler3_h)

        cortado = image[y0:y1, x0:x1]
        cortado = ocr(cortado)
        #imgplot = plt.imshow(cortado)

        information.append(cortado)
        i += 1

    print('Informacion general' + str(information))



if __name__ == "__main__":
    convert_img_to_array(
        'information.txt', '2f05e196-c6b2-4694-953c-daa69846cb1d.jpg')
