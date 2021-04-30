"""
Neato

MIT License

Copyright (c) 2021 Joseph Pace

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
