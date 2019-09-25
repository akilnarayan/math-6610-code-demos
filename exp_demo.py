
import numpy as np

def myexp(x, n):
    y = 1.
    for r in range(1,n):
        y += x**r/np.math.factorial(r)

    return y
