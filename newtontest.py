import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time


## Use numpy.power
def derivative(z):
    f = np.multiply(z, z)
    g = np.multiply(z, f)
    der = 4 * g
    return der

def poly(z):
    f = np.multiply(z, z)
    g = np.multiply(f, f)
    h = g - 1
    return h

def findSolutions(func):
    return np.array([1,-1, 1j,-1j])

def newton(z):
    z_next = z - poly(z)/derivative(z)
    return z_next



if __name__ == '__main__':


    tic = time.perf_counter()
    
    solutions = findSolutions(0)

    epsilon = 1e-5
    tol = 1e-4
    maxiter = 100
    xmin = 0
    xmax = 6
    xn = 1000
    ymin = -5
    ymax = 6
    yn = 1000


    X = np.linspace(xmin, xmax, xn).astype(np.float32)
    Y = np.linspace(ymin, ymax, yn).astype(np.float32)

    C = X + Y[:, None] * 1j

    colours = np.zeros_like(C, dtype = int)


    stillChecking = np.ones_like(C, dtype = bool)

    
    values = np.array(C)
    values_next = np.array(C)



    for i in range(maxiter):
        values_next[stillChecking] = newton(values[stillChecking])
        stillChecking = np.abs(values_next - values) > epsilon
        values = np.array(values_next)

    for i in range(len(values)):
        for j in range(len(values[i])):
            for k in range(len(solutions)):
                if (abs(solutions[k] - values[i][j]) < tol):
                    colours[i][j] = k + 1
                    break

    toc = time.perf_counter()

    print(f'Total time is {toc - tic: 05f} seconds')

    # print(np.sum(colours == 4))

    # print(np.sum(np.abs(values - 1j) < epsilon))
    # print(np.sum(np.abs(values - 1) < epsilon))
    # print(np.sum(np.abs(values + 1j) < epsilon))
    # print(np.sum(np.abs(values + 1) < epsilon))
