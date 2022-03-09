def run():
    LIMIT = 1000000
    contador = 0
    potencia_2 = 2**contador
    while potencia_2 < LIMIT:
        print("2 elevado a " + str(contador) + "Es igual a:" + str(potencia_2))
        contador = contador + 1
        potencia_2 = 2**contador



if __name__ == "__main__":
    run()