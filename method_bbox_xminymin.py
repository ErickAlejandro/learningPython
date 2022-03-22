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

    m = datos.pandas().xyxy[0]
    information = []

    # ordenar la matriz
    m = np.array(m)
    m = m[m[:, 0].argsort()]
    print(m)

    



if __name__ == "__main__":
    convert_img_to_array(
        'information.txt', '2f05e196-c6b2-4694-953c-daa69846cb1d.jpg')
