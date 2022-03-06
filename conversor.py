menu = """
Bienvenido al conversor de monedas ✔💫
Opcion1: Pesos Colombianos
Opcion2: Pesos Argentinos
Opcion3: Pesos Mexicanos

Elige una opcion: 
"""

def operations(tipo_pesos, valor_dolar):
    pesos = input("Por favor usted tiene pesos" + tipo_pesos +", ingrese el valor: ")
    pesos = float(pesos)
    dolares = pesos / valor_dolar
    dolares = round(dolares, 2)
    dolares = str(dolares)
    print("Tienes $ " + dolares + "dólares")

option = int(input(menu))

if option == 1:
    operations("Colombianos", 3875)
elif option == 2:
    operations("Argentinos", 65)
elif option == 3:
    operations("Mexicanos", 24)
else:
    print("Ingresa una opcion correcta.")

