import numpy as np
import matplotlib.pyplot as plt

n = 1000
it = 0
proj = "2D"
epsilon = 10e-6

dx = 1/n#*0.001

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)

u = np.zeros_like(x)
u[:] = 1
# u = -(x - 1/2)
# v = y - 1/2

dt = dx/max(abs(u))

phi = np.zeros_like(x)
initX = [x[int(n/4)], x[int(3*n/4)]]

def reinit(phi, init = False):
    if not init:
        init = []
        for i in range(len(phi)):
            if round(phi[i], 4) == 0:
                init.append(x[i])
        if init == []:
            init.append(0)

    for i in range(len(x)):
        # levelSet[i] = min([abs(x[i] -  initX[0]), abs(x[i] - initX[1])])
        phi[i] = min(abs(x[i] -  init))
        if len(init) > 1:
            if x[i] > init[0] and x[i] < init[1]:
                phi[i] = - phi[i]
    
    return phi

phi = reinit(phi, initX)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.plot(x, phi, 'k--', label='Signed distance function')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(initX, [0, 0], 'kx', label='Interface', markersize=10)
plt.plot([0, initX[0]], [0, 0], 'r', label='Fluid domain', markersize=8)
plt.plot([initX[1], 1], [0, 0], 'r', label='', markersize=8)
plt.plot([initX[0], initX[1]], [0, 0], 'b', label='Solid domain', markersize=8)
plt.xlabel('x', fontsize=14, style='italic')
plt.ylabel('y', fontsize=14, style='italic')
plt.legend(loc='upper right', fontsize=14)
plt.savefig('figures/SDFNew.png', dpi=900)

plt.show()
temp = np.zeros_like(phi)

for i in range(it):
    if i%5 == 0:
        phi = reinit(phi)
    for j in range(1, len(phi)-1):
        # temp[j] = phi[j] - u[j]*dt/dx*(phi[j] - phi[j-1]) # upwind
        temp[j] = phi[j] - u[j]*dt/(2*dx)*(phi[j+1] - phi[j-1]) # central
    temp[0] = phi[0] + (u[0] + epsilon)/abs(u[0] + epsilon)*(phi[0] - phi[1])
    temp[-1] = phi[-1] + (u[0] + epsilon)/abs(u[-1] + epsilon)*(phi[-2] - phi[-1])
    phi = temp
    plt.plot(x, phi)
    x1 = [initX[0], initX[1]]
    y1 = [0, 0]
    plt.plot(initX, [0, 0], 'x', markersize=10)
    plt.plot(initX, [0, 0])
    plt.title(i)

    plt.show()

if __name__ == "__main__":
    print(dt)