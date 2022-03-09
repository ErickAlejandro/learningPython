from cProfile import run


def run():
    pass

i = 0
nombre = input("Agrega tu nombre aqui: ")

while i < 900:
    i +=1
    print("Hola" + nombre + "estas en la posicion: " + str(i))
    
    if i == 120:
        break



if __name__ == "__main__":
    run()