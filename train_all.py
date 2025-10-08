from data_gen import generate_operators
from train_digits import train_digits
from train_operators import train_operators

def main():
    generate_operators(1000)
    train_digits()
    train_operators()
    print("Training complete")

if __name__ == "__main__":
    main()