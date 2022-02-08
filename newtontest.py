import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import sympy as sym

### stub, potentially use other lib for finding derivative
##def derivative(z):
##    ## Use numpy.power
##    f = np.multiply(z, z)
##    g = np.multiply(z, f)
##    der = 4 * g
##    return der
##
### stub, need to read function
##def poly(z):
##    f = np.multiply(z, z)
##    g = np.multiply(f, f)
##    h = g - 1
##    return h
##
### stub, needs to find solutions of func (to use with poly)
##def findSolutions(func):
##    return np.array([1,-1, 1j,-1j])
##
### Does one iteration of Newton-Raphson
def newton(z):
    z_next = z - poly(z)/der(z)
    return z_next
##
##def getValue(func, sub):
##    return parse_expr(func).subs(z, sub).evalf()



if __name__ == '__main__':

    z = sym.Symbol('z')
    func = input("Input a polynomial in z: ")
    func2 = sym.diff(func, z)

    poly = sym.lambdify(z, func)
    der = sym.lambdify(z, func2)
    
    
    # Parameters

    tic = time.perf_counter()
    
##    solutions = findSolutions(0)
    solutions = sym.solve(func)
    print(solutions)
    
    for i in range(len(solutions)):
        solutions[i] = complex(solutions[i])


    

    epsilon = 1e-7
    tol = 1e-5
    maxiter = 100
    xmin = -5
    xmax = 5
    xn = 1000
    ymin = -5
    ymax = 5
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
##        stillChecking = np.abs(values_next - values) > epsilon
        stillChecking = np.abs(poly(values_next)) > epsilon
        values = np.array(values_next)

##    for i in range(len(values)):
##        for j in range(len(values[i])):
##            for k in range(len(solutions)):
##                if (abs(solutions[k] - values[i][j]) < tol):
##                    colours[i][j] = k + 1
##                    break


    # Using array operations is 5 times faster for f(z) = z**5 + z**2 - 0.1.
    for k in range(len(solutions)):
        I = abs(solutions[k] - values) < tol
        colours[I] = k + 1
        temp = np.sum(I)
        print(f'Total number of the {k}th colour is {temp}')

    print(np.sum(colours == 0))


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
