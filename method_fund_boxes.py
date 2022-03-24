from math import ceil
import pytesseract as pyt
import torch


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



def convert_img_to_array(img):
    # Toma la imagen desde el webservice
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='best.pt', device='cpu')
    results = model(img)
    results = results.pandas().xyxy[0]
    
    m = results.to_numpy() #Transformar la matrix panda a numpy
    print(m)
    print(type(m))
    
    information = []

    # Extraer las dimensiones de la imagen a usar
    y, x = img.shape[:2]
    print('\nEl ancho: ' + str(x) + ' y el alto: ' + str(y) + '\n')

    # Tranformacion de pixel para ingresar los datos en el OCR
    i = 0
    for i in range(len(m)):
        print('Fila de contador numero ' + str(i))
        xmin = m[i, 0]
        ymin = m[i, 1]
        xmax = m[i, 2]
        ymax = m[i, 3]
        print('Punto Xmin: ' + str(xmin) + '\nPunto Ymin: ' + str(
            ymin) + '\nXmax: ' + str(xmax) + '\nymax: ' + str(ymax) + '\n')

        x0 = ceil(xmin)
        y0 = ceil(ymin)

        x1 = ceil(xmax)
        y1 = ceil(ymax)

        print(x0, y0, x1, y1)
        
        cortado = img[x0:x1, y0:y1]
        cut = ocr(cortado)

        information.append(cut)
        i += 1

    print('Informacion general\n' + str(information))


if __name__ == "__main__":
    convert_img_to_array()
