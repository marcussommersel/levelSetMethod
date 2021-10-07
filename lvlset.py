import numpy as np
import matplotlib.pyplot as plt

n = 10

nx = 10
ny = nx
x = np.linspace(0, nx, n)
y = np.linspace(0, ny, n)
F = 0.01 # some function
phi = np.random.randn(n, n) # initial value for phi
dt = 1
it = 100

for i in range(it):
    dphi = np.gradient(phi)
    dphi_norm = np.sqrt(np.sum(np.square(dphi), axis=0))

    phi = phi + dt * F * dphi_norm

    # plot the zero level curve of phi
    plt.contour(phi, 0)
    plt.title(i)
    plt.pause(0.01)
    plt.clf()
    # plt.show()