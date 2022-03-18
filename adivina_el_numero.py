import random as rd

def run():
    numero_aleatorio = rd.randint(1, 100)
    numero = int(input('Intenta adivinar un numero: '))

    while numero != numero_aleatorio:
        if numero <  numero_aleatorio:
            print('busca un numero mas alto :c')
        elif numero > numero_aleatorio:
            print('Busca un numero menor :c')
        numero = int(input('Elige otro numero: '))

    print('Adivinaste! ')

if __name__ == '__main__':
    run()