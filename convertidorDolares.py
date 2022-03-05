cantidad = input("Cuanta plata quieres convertir ?: ")
cantidad = float(cantidad)
dolar = 683
calculo = cantidad / dolar
calculo = round(calculo, 3)
calculo = str(calculo)
print("Tienes aproximadamente " + calculo + " Dolares")