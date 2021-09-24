import numpy as np


def print_const(arr):
    print(repr(arr[1]))


def main():
    x = np.array([.1, .2, .9])
    print_const(x)
    x += 1
    print_const(x)


if __name__ == "__main__":
    main()
