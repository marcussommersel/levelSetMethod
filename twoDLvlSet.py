import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
import time

def init(phi, init, c, r):

    p = path.Path(np.transpose(init))
    for i in range(len(x)):
        for j in range(len(y)):
            if p.contains_points([(x[i], y[j])]):
                phi[i,j] = - min(np.sqrt((x[i] - init[0])**2 + (y[j] - init[1])**2))
            else:
                phi[i,j] = min(np.sqrt((x[i] - init[0])**2 + (y[j] - init[1])**2))
            # if (x[i] - c[0])**2 + (y[j] - c[1])**2 < r**2:
            #     phi[i,j] = - min(np.sqrt((x[i] - init[0])**2 + (y[j] - init[1])**2))
            # else:
            #     phi[i,j] = min(np.sqrt((x[i] - init[0])**2 + (y[j] - init[1])**2))
    return phi

def TVDRK3(phi, scheme, u, v):

    # first euler step
    phix, phiy = scheme(phi, u, v)
    n1 = phi - dt*(u*phix + v*phiy)

    # second euler step
    phix, phiy = scheme(n1, u, v)
    n2 = n1 - dt*(u*phix + v*phiy)

    # averaging step
    n1_2 = 3/4*phi + 1/4*n2

    # third euler step
    phix, phiy = scheme(n1_2, u, v)
    n3_2 = n1_2 - dt*(u*phix + v*phiy)

    # second averaging step
    temp =  1/3*phi + 2/3*n3_2

    if (scheme) == weno:
        temp = wenoBC(temp)

    return temp

def euler(phi, scheme, u, v):
    phix, phiy = scheme(phi, u, v)
    temp = phi - dt*(u*phix + v*phiy)

    if (scheme) == weno:
        temp = wenoBC(temp)

    return temp

def upwind(phi, u, v):
    phix = np.zeros([len(x), len(y)])
    phiy = np.zeros([len(x), len(y)])
    for i in range(2, len(x)-2):
        for j in range(2, len(y)-2):

            if u[i,j] >= 0:
                phix[i, j] = (phi[i, j] - phi[i - 1, j])/dx
            elif u[i,j] < 0:
                phix[i, j] = (phi[i + 1, j] - phi[i, j])/dx

            if v[i,j] >= 0:
                phiy[i, j] = (phi[i, j] - phi[i, j - 1])/dy
            elif v[i,j] < 0:
                phiy[i, j] = (phi[i, j + 1] - phi[i, j])/dy

    return phix, phiy

def upwind2(phi, u, v):
    phix = np.zeros([len(x), len(y)])
    phiy = np.zeros([len(x), len(y)])
    for i in range(2, len(x)-2):
        for j in range(2, len(y)-2):

            if u[i, j] >= 0:
                phix[i, j] = (3*phi[i,j] - 4*phi[i-1, j] + phi[i-2, j])/(2*dx)
            elif u[i, j] < 0:
                phix[i, j] = (-phi[i+2, j] + 4*phi[i+1, j] - 3*phi[i,j])/(2*dx)

            if v[i, j] >= 0:
                phiy[i, j] = (3*phi[i,j] - 4*phi[i, j-1] + phi[i, j-2])/(2*dx)
            elif v[i, j] < 0:
                phiy[i, j] = (-phi[i, j+2] + 4*phi[i, j+1] - 3*phi[i,j])/(2*dx)

    return phix, phiy

def central(phi, u, v):
    phix = np.zeros([len(x), len(y)])
    phiy = np.zeros([len(x), len(y)])
    for i in range(2, len(x)-2):
        for j in range(2, len(y)-2):

            phix[i, j] = (phi[i + 1, j] - phi[i - 1, j])/(2*dx)
            phiy[i, j] = (phi[i, j + 1] - phi[i, j - 1])/(2*dy)

    return phix, phiy

def weno(phi, u, v):
    phix = np.zeros([len(x), len(y)])
    phiy = np.zeros([len(x), len(y)])
    for i in range(2, len(x)-3):
        for j in range(2, len(y)-3):

            # WENO
            if u[i,j] >= 0:
                v1 = (phi[i-2,j] - phi[i-3,j])/dx
                v2 = (phi[i-1,j] - phi[i-2,j])/dx
                v3 = (phi[i,j] - phi[i-1,j])/dx
                v4 = (phi[i+1,j] - phi[i,j])/dx
                v5 = (phi[i+2,j] - phi[i+1,j])/dx
            elif u[i,j] < 0:
                v1 = (phi[i-1,j] - phi[i-2,j])/dx
                v2 = (phi[i,j] - phi[i-1,j])/dx
                v3 = (phi[i+1,j] - phi[i,j])/dx
                v4 = (phi[i+2,j] - phi[i+1,j])/dx
                v5 = (phi[i+3,j] - phi[i+2,j])/dx

            S1 = 13/12*(v1 - 2*v2 + v3)**2 + 1/4*(v1 - 4*v2 + v3)**2
            S2 = 13/12*(v2 - 2*v3 + v4)**2 + 1/4*(v2 - v4)**2
            S3 = 13/12*(v3 - 2*v4 + v5)**2 + 1/4*(3*v3 - 4*v4 + v5)**2

            epsilon = 10**-6*max(v1**2, v2**2, v3**2, v4**2, v5**2) + 10**-99

            alpha1 = 0.1/(S1 + epsilon)**2
            alpha2 = 0.6/(S2 + epsilon)**2
            alpha3 = 0.3/(S3 + epsilon)**2

            omega1 = alpha1/(alpha1 + alpha2 + alpha3)
            omega2 = alpha2/(alpha1 + alpha2 + alpha3)
            omega3 = alpha3/(alpha1 + alpha2 + alpha3)

            phix1 = v1/3 - 7*v2/6 + 11*v3/6
            phix2 = -v2/6 + 5*v3/6 + v4/3
            phix3 = v3/3 + 5*v4/6 - v5/6

            phix[i,j] = (omega1*phix1 + omega2*phix2 + omega3*phix3)

            if v[i,j] >= 0:
                v1 = (phi[i,j-2] - phi[i,j-3])/dx
                v2 = (phi[i,j-1] - phi[i,j-2])/dx
                v3 = (phi[i,j] - phi[i,j-1])/dx
                v4 = (phi[i,j+1] - phi[i,j])/dx
                v5 = (phi[i,j+2] - phi[i,j+1])/dx
            elif u[i,j] < 0:
                v1 = (phi[i,j-1] - phi[i,j-2])/dx
                v2 = (phi[i,j] - phi[i,j-1])/dx
                v3 = (phi[i,j+1] - phi[i,j])/dx
                v4 = (phi[i,j+2] - phi[i,j+1])/dx
                v5 = (phi[i,j+3] - phi[i,j+2])/dx

            S1 = 13/12*(v1 - 2*v2 + v3)**2 + 1/4*(v1 - 4*v2 + v3)**2
            S2 = 13/12*(v2 - 2*v3 + v4)**2 + 1/4*(v2 - v4)**2
            S3 = 13/12*(v3 - 2*v4 + v5)**2 + 1/4*(3*v3 - 4*v4 + v5)**2

            epsilon = 10**-6*max(v1**2, v2**2, v3**2, v4**2, v5**2) + 10**-99

            alpha1 = 0.1/(S1 + epsilon)**2
            alpha2 = 0.6/(S2 + epsilon)**2
            alpha3 = 0.3/(S3 + epsilon)**2

            omega1 = alpha1/(alpha1 + alpha2 + alpha3)
            omega2 = alpha2/(alpha1 + alpha2 + alpha3)
            omega3 = alpha3/(alpha1 + alpha2 + alpha3)

            phiy1 = v1/3 - 7*v2/6 + 11*v3/6
            phiy2 = -v2/6 + 5*v3/6 + v4/3
            phiy3 = v3/3 + 5*v4/6 - v5/6

            phiy[i,j] = (omega1*phiy1 + omega2*phiy2 + omega3*phiy3)

    return phix, phiy

def wenoBC(fun):

    fun[2, :] = fun[3, :] - (fun[4, :]- fun[3, :])
    fun[1, :] = fun[2, :] - (fun[3, :]- fun[2, :])
    fun[0, :] = fun[1, :] - (fun[2, :]- fun[1, :])
    fun[-3, :] = fun[-4, :] - (fun[-5, :]- fun[-4, :])
    fun[-2, :] = fun[-3, :] - (fun[-4, :]- fun[-3, :])
    fun[-1, :] = fun[-2, :] - (fun[-3, :]- fun[-2, :])

    fun[:, 2] = fun[:, 3] - (fun[:, 4]- fun[:, 3])
    fun[:, 1] = fun[:, 2] - (fun[:, 3]- fun[:, 2])
    fun[:, 0] = fun[:, 1] - (fun[:, 2]- fun[:, 1])
    fun[:, -3] = fun[:, -4] - (fun[:, -5]- fun[:, -4])
    fun[:, -2] = fun[:, -3] - (fun[:, -4]- fun[:, -3])
    fun[:, -1] = fun[:, -2] - (fun[:, -3]- fun[:, -2])

    return fun

def reinit(phi, scheme, u, v):
    # temp = phi
    dtau = 0.5*dx
    S0 = phi/(np.sqrt(phi**2 + max(dx, dy)**2))
    # S0 = phi/(np.sqrt(phi**2 + 2*dx**2)) # from Karl Yngve Lervåg (2013)
    for k in range(tmax):

        # TVDRK3

        temp = np.zeros_like(phi)
        n1 = np.zeros_like(phi)
        n2 = np.zeros_like(phi)
        n1_2 = np.zeros_like(phi)
        n3_2 = np.zeros_like(phi)

        # first euler step
        phix, phiy = scheme(phi, u, v)
        S = phi/np.sqrt(phi**2 + abs(phix + phiy)**2*max(dx, dy)**2)
        n1 = phi - dtau*S*(np.sqrt(phix**2 + phiy**2) - 1)

        # second euler step
        phix, phiy = scheme(n1, u, v)
        S = n1/np.sqrt(n1**2 + abs(phix + phiy)**2*max(dx, dy)**2)
        n2 = n1 - dtau*S*(np.sqrt(phix**2 + phiy**2) - 1)

        # averaging step
        n1_2 = 3/4*phi + 1/4*n2

        # third euler step
        phix, phiy = scheme(n1_2, u, v)
        S = n1_2/np.sqrt(n1_2**2 + abs(phix + phiy)**2*max(dx, dy)**2)
        n3_2 = n1_2 - dtau*S*(np.sqrt(phix**2 + phiy**2) - 1)

        # second averaging step
        temp = 1/3*phi + 2/3*n3_2

        # temp =  phi - dt*S0*(np.sqrt((u*phix)**2 + (v*phiy)**2) - 1)

        temp = wenoBC(temp)
        phi = temp
        # plottingContour()

        phix, phiy = scheme(phi, u, v)
        phiGradAbs = abs(phix + phiy)

    return phi, phiGradAbs

def plottingContour(title = ''):
    if proj == '2D':
        plt.plot(initX[0], initX[1], 'r')
        plt.contourf(x, y, np.transpose(phi), 0)
        plt.colorbar()
        # plt.ylim(0, 1)
        # plt.xlim(0, 1)
    elif proj == '3D':
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, phi, 50, cmap='coolwarm')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    n = 64
    tmax = 1 # number of timesteps in reinitialization
    it = 101
    proj = "2D"
    testCase = 'zalesak'
    epsilon = 10e-6

    dx = 1/n
    dy = 1/n

    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)

    u = np.zeros([len(x), len(y)])
    v = np.zeros([len(x), len(y)])
    # u[:,:] = 1
    # v[:,:] = 1

    phi = np.zeros([len(x), len(y)])

    # initX = [a*np.cos(theta) + 0.5, a*np.sin(theta) + 0.75]

    totalTime = 0
    dt = 0
    t = 0
    T = 1

    def uVortex(i,j):
        return -2*(np.sin(np.pi*x[i]))**2*np.sin(np.pi*y[j])*np.cos(np.pi*y[j])*np.cos(np.pi*t/T)

    def vVortex(i,j):
        return -2*np.sin(np.pi*x[i])*np.cos(np.pi*x[i])*(np.cos(np.pi*y[j]))**2*np.cos(np.pi*t/T)

    def uZalesak(i,j):
        return -np.pi/628*(x[i]**2 + 2*y[j] - x[i] - 1)

    def vZalesak(i,j):
        return np.pi/628*(2*x[i] + y[j]**2 - 1 - y[j])

    # initial boundary
    cx = 0.5 # center of initial boundary
    cy = 0.75
    a = 0.15 # radius
    theta = np.linspace(0, 2*np.pi, n)
    if testCase == 'vortex':
        uvel = uVortex
        vvel = vVortex
        initX = [a*np.cos(theta) + 0.5, a*np.sin(theta) + 0.75]
    elif testCase == 'zalesak':
        uvel = uZalesak
        vvel = vZalesak

        width = 0.05
        length = 0.25 # values from Claudio Walker
        thetaZ1 = np.linspace(0, 3/2*np.pi - np.arcsin(0.5*width/a), n)
        thetaZ2 = np.linspace(3/2*np.pi + np.arcsin(0.5*width/a), 2*np.pi, n)
        x1 = -width/2 + cx
        x2 = width/2 + cx
        y1 = -np.sqrt(a**2 - width**2) + cy 
        y2 = y1 + length
        xinit = a*np.cos(thetaZ1) + cx
        xinit = np.append(xinit, [x1, x2])
        xinit = np.append(xinit, [a*np.cos(thetaZ2) + cx])

        yinit = a*np.sin(thetaZ1) + cy
        yinit = np.append(yinit, [y2, y2])
        yinit = np.append(yinit, [a*np.sin(thetaZ2) + cy])
        initX = [xinit, yinit]

    phi = init(phi, initX, [cx, cy], a)
    phi0 = phi

    # Godunov, boken til sethien.

    for k in range(it):
        if k == 0:
            plottingContour("t = " + str(t) + ", it = " + str(k) + ", t/T = " + str(t/T))

        startTime = time.time()

        u = np.fromfunction(uvel, (len(x), len(y)), dtype=int)
        v = np.fromfunction(vvel, (len(x), len(y)), dtype=int)

        # CFL condition
        dt = 0.5*(dx + dy)/(u + v).max()
        t += dt

        if k%1 == 0 and k != 0:
            phi, dPhi = reinit(phi, weno, u, v)

        phi = TVDRK3(phi, weno, u, v)

        currentTime = time.time() - startTime
        totalTime += currentTime # plotting not included
        print("iteration = " + str(k) + ", time = " + str(t) + ", iteration time = " + str(totalTime) + ", t/T = " + str(t/T))
        if k%1 == 0 and k != 0:
            plottingContour("t = " + str(k*dt) + ", it = " + str(k) + ", t/T = " + str(t/T))