menu = """
Bienvenido al conversor de monedas ✔💫
Opcion1: Pesos Colombianos
Opcion2: Pesos Argentinos
Opcion3: Pesos Mexicanos

Elige una opcion: 
"""

option = int(input(menu))

if option == 1:
    pesos = input("Por favor, ingrese el valor: ")
    pesos = float(pesos)
    valor_dolar = 3875
    dolares = pesos / valor_dolar
    dolares = round(dolares, 2)
    dolares = str(dolares)
    print("Tienes $ " + dolares + "dólares")
elif option == 2:
    pesos = input("Por favor, ingrese el valor: ")
    pesos = float(pesos)
    valor_dolar = 65
    dolares = pesos / valor_dolar
    dolares = round(dolares, 2)
    dolares = str(dolares)
    print("Tienes $ " + dolares + "dólares")
elif option == 3:
    pesos = input("Por favor, ingrese el valor: ")
    pesos = float(pesos)
    valor_dolar = 24
    dolares = pesos / valor_dolar
    dolares = round(dolares, 2)
    dolares = str(dolares)
    print("Tienes $ " + dolares + "dólares")
else:
    print("Ingresa una opcion correcta.")

