def run():
    # for contador in range(0, 1000):
    #     if contador % 2 != 0:
    #         continue 
    #     print(contador)


    # for i in range(0, 10000):
    #     print(i)
    #     if i == 5678:
    #         break

    text = input("Selecciona una palabra: ")
    for letra in text:
        if letra == "o":
            break
        print(letra)


if __name__ == "__main__":
    run()