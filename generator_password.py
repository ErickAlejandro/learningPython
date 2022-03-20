import random as rd


def generate_password():
    mayus = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    minus = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numeros = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    simbols = ['?', '#', '%', '/', '(', '*']

    characters = mayus + minus + numeros + simbols
    password = []

    for i in range(15):
        characters_random = rd.choice(characters)
        password.append(characters_random)
    password = ''.join(password)
    return password

def run():
    password = generate_password()
    print('Your new password is: ' +  password)

if __name__ == '__main__':
    run()