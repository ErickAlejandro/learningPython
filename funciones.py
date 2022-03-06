
# def imprimir_mensaje():
#     print("Mensaje especial:")
#     print("I'm learning to use functions")

# imprimir_mensaje()

def conversacion (mensaje):
    print("Hola")
    print("Como estas ?")
    print("Elige una opcion" + mensaje)
    print("Adios")

option = int(input("Elige una opcion (1, 2, 3): "))

if option == 1:
    conversacion(str(option))
elif option == 2:
    conversacion(str(option))
elif option == 3:
    conversacion(str(option))
else:
    print("Escribe la opcion correcta")