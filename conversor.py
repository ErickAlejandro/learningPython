pesos = input("Por favor, ingrese el valor: ")
pesos = float(pesos)
valor_dolar = 3975
dolares = pesos / valor_dolar
dolares = round(dolares, 2)
dolares = str(dolares)
print("Tienes $ " + dolares + "dólares")