import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time

# stub, potentially use other lib for finding derivative
def derivative(z):
    ## Use numpy.power
    f = np.multiply(z, z)
    g = np.multiply(z, f)
    der = 4 * g
    return der

# stub, need to read function
def poly(z):
    f = np.multiply(z, z)
    g = np.multiply(f, f)
    h = g - 1
    return h

# stub, needs to find solutions of func (to use with poly)
def findSolutions(func):
    return np.array([1,-1, 1j,-1j])

# Does one iteration of Newton-Raphson
def newton(z):
    z_next = z - poly(z)/derivative(z)
    return z_next



if __name__ == '__main__':

    # Parameters

    tic = time.perf_counter()
    
    solutions = findSolutions(0)

    epsilon = 1e-5
    tol = 1e-4
    maxiter = 100
    xmin = -3
    xmax = 3
    xn = 1000
    ymin = -3
    ymax = 3
    yn = 1000

    # Create complex plane
    X = np.linspace(xmin, xmax, xn).astype(np.float32)
    Y = np.linspace(ymin, ymax, yn).astype(np.float32)
    C = X + Y[:, None] * 1j

    # Check if complex point has converged onto a solution
    isConvergent = np.zeros_like(C, dtype = bool)
    colours = np.zeros_like(C, dtype = int)


    # Check if consecutive NR sequence converges
    stillChecking = np.ones_like(C, dtype = bool)
    # remove division by zero case, fix later
    stillChecking[0,0] = False

    
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

    dpi = 72
    width = 10
    height = 10*yn/xn
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
    cmap = colors.ListedColormap(['k','b','y','g','r'])

    # Shaded rendering
    light = colors.LightSource(azdeg=315, altdeg=10)
    M = light.shade(colours, cmap=cmap, vert_exag=1.5,
                    blend_mode='hsv')
    ax.imshow(M, extent=[xmin, xmax, ymin, ymax], interpolation="bicubic")
    ax.set_xticks([])
    ax.set_yticks([])

    toc = time.perf_counter()
    plt.show()


    print(f'Total time is {toc - tic: 05f} seconds')

    # print(np.sum(colours == 4))

    # print(np.sum(np.abs(values - 1j) < epsilon))
    # print(np.sum(np.abs(values - 1) < epsilon))
    # print(np.sum(np.abs(values + 1j) < epsilon))
    # print(np.sum(np.abs(values + 1) < epsilon))
