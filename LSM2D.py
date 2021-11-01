import schemes as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
import time

def init(phi, init):
    p = path.Path(np.transpose(init))
    for i in range(len(x)):
        for j in range(len(y)):
            if p.contains_points([(x[i], y[j])]):
                phi[i,j] = - min(np.sqrt((x[i] - init[0])**2 + (y[j] - init[1])**2))
            else:
                phi[i,j] = min(np.sqrt((x[i] - init[0])**2 + (y[j] - init[1])**2))
    return phi

def reinit(phi, scheme):
    dtau = 0.5*dx
    # S0 = phi/(np.sqrt(phi**2 + 2*dx**2)) # from Karl Yngve Lervåg (2013)
    # S = S0
    for k in range(tmax):

        # TVDRK3
        temp = np.zeros_like(phi)
        n1 = np.zeros_like(phi)
        n2 = np.zeros_like(phi)
        n1_2 = np.zeros_like(phi)
        n3_2 = np.zeros_like(phi)

        S0 = phi/(np.sqrt(phi**2 + max(dx, dy)**2))
        S = S0

        # first euler step
        phix, phiy = scheme(phi, S, S, x, y, dx, dy)
        S = phi/np.sqrt(phi**2 + abs(phix + phiy)**2*max(dx, dy)**2)
        n1 = phi - dtau*S*(np.sqrt(phix**2 + phiy**2) - 1)

        # second euler step
        phix, phiy = scheme(n1, S, S, x, y, dx, dy)
        S = n1/np.sqrt(n1**2 + abs(phix + phiy)**2*max(dx, dy)**2)
        n2 = n1 - dtau*S*(np.sqrt(phix**2 + phiy**2) - 1)

        # averaging step
        n1_2 = 3/4*phi + 1/4*n2

        # third euler step
        phix, phiy = scheme(n1_2, S, S, x, y, dx, dy)
        S = n1_2/np.sqrt(n1_2**2 + abs(phix + phiy)**2*max(dx, dy)**2)
        n3_2 = n1_2 - dtau*S*(np.sqrt(phix**2 + phiy**2) - 1)

        # second averaging step
        temp = 1/3*phi + 2/3*n3_2

        # temp =  phi - dt*S0*(np.sqrt((u*phix)**2 + (v*phiy)**2) - 1)

        temp = sc.wenoBC(temp)
        phi = temp

        S = phi/np.sqrt(phi**2 + abs(phix + phiy)**2*max(dx, dy)**2)
        phix, phiy = scheme(phi, S, S, x, y, dx, dy)
        phiGradAbs = abs(phix + phiy)

    return phi

def area(c):
    a = 0 
    x0,y0 = c[0]
    for [x1,y1] in c[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return a

def plottingContour(title = '', save=False, limitx=[-1,1], limity=[-1,1]):
    if proj == '2D':
        plt.plot(initX[0], initX[1], 'b', label='Initial interface')
        c = plt.contour(x, y, np.transpose(phi), 0, cmap='gray')
        # plt.colorbar(c)
        plt.axis('equal')
        plt.xlim(limitx)
        plt.ylim(limity)
        plt.xlabel('x')
        plt.ylabel('y')
        p = c.collections[1].get_paths()[0]
        vs = p.vertices
        a = area(vs)
        print('Area = {0} for {1}'.format(a, title))
    elif proj == '3D':
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, phi, 50, cmap='coolwarm')
    plt.title(title)
    if save:
        plt.savefig(('{0}, Reinit = {1} at freq = {2}, n = {3}, {4}.png'.format(title, tmax, reinitfreq, n, testCase)))
    plt.show()
    return a

if __name__ == '__main__':
    n = 128
    tmax = 10 # number of timesteps in reinitialization
    reinitfreq = 20
    doreinit = False
    dosave = True
    it = 10001

    CFL = 0.25

    proj = '2D'
    testCase = 'vortex'
    T = 2 # used in vortex-test
    
    dx = 1/n
    dy = 1/n

    x = np.linspace(0, 1, n)
    y = np.linspace(0.5, 1.5, n)
    u = np.zeros([len(x), len(y)])
    v = np.zeros([len(x), len(y)])
    phi = np.zeros([len(x), len(y)])

    totalTime = 0
    dt = 0
    t = 0

    def uVortex(i,j):
        return -2*((np.sin(np.pi*x[i]))**2)*np.sin(np.pi*y[j])*np.cos(np.pi*y[j])*np.cos(np.pi*t/T)
    def vVortex(i,j):
        return -2*np.sin(np.pi*x[i])*np.cos(np.pi*x[i])*((np.cos(np.pi*y[j]))**2)*np.cos(np.pi*t/T)
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

        dt = 0.0025
        plotcriteria = T

        uvel = uVortex
        vvel = vVortex
        initX = [a*np.cos(theta) + 0.5, a*np.sin(theta) + 0.75]
    elif testCase == 'zalesak':

        dt = 0.25
        plotcriteria = 628

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
    elif testCase == 'pospos':

        dt = 0.25
        plotcriteria = 1

        u[:,:] = -1
        v[:,:] = -1
        a = 0.5
        initX = [a*np.cos(theta), a*np.sin(theta)]

    phi = init(phi, initX)

    for k in range(it):

        print('iteration = {0}, time = {1:.5f}, iteration time = {2:.2f}, t/T = {3:.5f}'.format(k, t, totalTime, t/T))
        if k%10 == 0 or round(t/plotcriteria, 3) == 1.00 or round(t/plotcriteria, 3) == 0.50:
            a = plottingContour('t = {0:.2f}, it = {1}'.format(t, k), dosave, [x[0],x[-1]], [y[0],y[-1]])
            if k == 0:
                initialArea = a
            else:
                print('Area change = ' + str(100*initialArea/a-100) + '%')

        if k%reinitfreq == 0 and k != 0 and doreinit:
            reinitStart = time.time()
            phi = reinit(phi, sc.godunov)
            reinitTime = time.time() - reinitStart
            print('Reinitialization time = {0}'.format(reinitTime))
            totalTime += reinitTime
            plottingContour('t = {0:.2f}, it = {1}, reinit'.format(k*dt, k), [x[0],x[-1]], [y[0],y[-1]])

        startTime = time.time()

        if not testCase == 'pospos':
            u = np.fromfunction(uvel, (len(x), len(y)), dtype=int)
            v = np.fromfunction(vvel, (len(x), len(y)), dtype=int)

        # CFL condition
        dtmax = CFL*(dx + dy)/(abs(u + v)).max()
        if dt > dtmax:
            print('WARNING; dt too high at it = {0}, dt = {1}, dtmax = {2}'.format(k, dt, dtmax))
            break
        # dt =5*10**-5 # From Claudio Walker article [a]
        # dt = 10**-3
        t += dt

        phi = sc.TVDRK3(phi, sc.weno, u, v, x, y, dx, dy, dt)

        currentTime = time.time() - startTime
        totalTime += currentTime # plotting not included
