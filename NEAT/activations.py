
import math
import random


# --- Activation functions ---

def absolute(x): return abs(x)


def gaussian(x): return math.e ** -(x * x)


def linear(x): return x


def modified_sigmoid(x): return 1 / (1 + math.e ** (-4.9 * x))


def modified_tanh(x): math.tanh(-2.45 * x)


def relu(x): return max(0.0, x)


def sigmoid(x): return 1 / (1 + math.e ** -x)


def sine(x): return math.sin(x)


def square(x): return x * x


def srelu(x):
    if x <= 0.001:
        return 0.001 + (x - 0.001) * 0.00001
    elif x >= 0.999:
        return 0.999 + (x - 0.999) * 0.00001
    else:
        return x


def tanh(x): return math.tanh(x)


# --- Other functions ---

def get_random():
    return random.choice([absolute, gaussian, linear, modified_sigmoid, modified_tanh, relu, sigmoid, sine, square, srelu, tanh])
