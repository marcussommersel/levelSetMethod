import schemes as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
import time
from skimage import measure

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

        # S = phi/np.sqrt(phi**2 + abs(phix + phiy)**2*max(dx, dy)**2)
        # phix, phiy = scheme(phi, S, S, x, y, dx, dy)
        # phiGradAbs = abs(phix + phiy)

    return phi

def area(xval, yval):
    a = 0
    x1 = xval/(n-1)
    y1 = yval/(n-1)
    # x1 = c[:, 0]/(n-1)
    # y1 = c[:, 1]/(n-1)
    for i in range(len(x1)-1):
        dx = x1[i+1]-x1[i]
        dy = y1[i+1]-y1[i]
        a += 0.5*(y1[i]*dx - x1[i]*dy)
    return abs(a)

def contour(fun): 
    c = measure.find_contours(fun, 0)
    if len(c) > 1: # Fix for cases where function finds several contours
        c = c[0]
        # c = np.append(c[0], c[1])
        # if c.dtype == 'float64':
        #     break
    con = np.asarray(c, dtype='float64').reshape(-1)

    #     con = np.asarray(con).reshape(-1)
    xval = con[0::2]   
    yval = con[1::2]
    return xval, yval
    # return np.array(c).ravel()
    # con = np.squeeze(np.asarray(c))
    # return con

def plottingContour(title = '', case='', save=False, limitx=[-1,1], limity=[-1,1]):
    if case == 'vortex':
        yoffset = 0.25
    else:
        yoffset = 0
    if proj == '2D':
        plt.plot(initX[0], initX[1], 'b', label='Initial interface')
        plt.axis('equal')
        plt.xlim(limitx)
        plt.ylim(limity)
        plt.xlabel('x')
        plt.ylabel('y')
        xval, yval = contour(phi)
        
        # plt.plot(con[:, 0]/(n-1), con[:, 1]/(n-1) + yoffset, 'r', linewidth=2, label='Current interface')
        plt.plot(xval/(n-1), yval/(n-1) + yoffset, 'r', linewidth=2, label='Current interface')
        plt.legend()
    elif proj == '3D':
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, phi, 50, cmap='coolwarm')
    plt.title(title)
    if save:
        if doreinit:
            plt.savefig(('figures/{0}, n = {1}, {2}, Reinit iter = {3} at freq = {4}.png'.format(title, n, testCase, tmax, reinitfreq)))
        else:
            plt.savefig(('figures/{0}, n = {1}, {2}, no Reinit.png'.format(title, n, testCase)))
    plt.show()

if __name__ == '__main__':
    n = 256
    tmax = 10 # number of timesteps in reinitialization
    reinitfreq = 50
    doreinit = True
    dosave = False
    it = 100001

    CFL = 0.25

    proj = '2D'
    testCase = 'zalesak'
    T = 2 # used in vortex-test

    dx = 1/n
    dy = 1/n

    x = np.linspace(0, 1, n)
    if testCase == 'vortex':
        y = np.linspace(0.25, 1.25, n)
    else:
        y = np.linspace(0, 1, n)
    u = np.zeros([len(x), len(y)])
    v = np.zeros([len(x), len(y)])
    phi = np.zeros([len(x), len(y)])

    totalTime = 0
    dt = 0
    t = 0

    def uVortex(i,j):
        return -2*((np.sin(np.pi*x[i]))**2)*np.sin(np.pi*y[j])*np.cos(np.pi*y[j])*np.cos(np.pi*t/T) # Claudio Walker
        # return -2*(np.sin(np.pi*x[i] - 0.5*np.pi))**2*np.cos(np.pi*y[j] - 0.5*np.pi)*np.sin(np.pi*y[j] - 0.5*np.pi)
    def vVortex(i,j):
        return -2*np.sin(np.pi*x[i])*np.cos(np.pi*x[i])*((np.cos(np.pi*y[j]))**2)*np.cos(np.pi*t/T) # Claudio Walker
        # return -2*(np.cos(np.pi*y[j] - 0.5*np.pi))**2*np.sin(np.pi*x[i] - 0.5*np.pi)*np.cos(np.pi*x[i] - 0.5*np.pi)
    def uZalesak(i,j):
        # return -np.pi/628*(x[i]**2 + 2*y[j] - x[i] - 1) # Claudio Walker
        return np.pi/10*(0.5 - y[j])
    def vZalesak(i,j):
        # return np.pi/628*(2*x[i] + y[j]**2 - 1 - y[j]) # Claudio Walker
        return np.pi/10*(x[i] - 0.5)

    theta = np.linspace(0, 2*np.pi, n)
    if testCase == 'vortex':

        # Claudio Walker:
        a = 0.15
        cx = 0.5
        cy = 0.75
        #dt = 0.0025
        dt = 0.0005
        plotcriteria = T

        uvel = uVortex
        vvel = vVortex
        
        initX = [a*np.cos(theta) + 0.5, a*np.sin(theta) + 0.75]
    elif testCase == 'zalesak':

        # Åsmund Ervik:
        cx = 0.5
        cy = 0.5
        dt = 0.005
        a = 1/3
        # plotcriteria = 628
        plotcriteria = 20

        uvel = uZalesak
        vvel = vZalesak

        width = a/3 # 0.05
        length = width*5 # 0.25 # values from Claudio Walker
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
    # con = contour(phi)
    # initialArea = area(con)
    xval, yval = contour(phi)
    initialArea = area(xval, yval)
    print("Initial area = {0}".format(initialArea))
    for k in range(it):

        if k%100 == 0:
            print('iteration = {0}, time = {1:.5f}, iteration time = {2:.2f}, t/T = {3:.5f}'.format(k, t, totalTime, t/T))
        if k%1000 == 0 or round(t/plotcriteria, 4) == 1.00 or round(t/plotcriteria, 4) == 0.25:
            title = 't = {0:.3f}, it = {1}'.format(t, k)
            plottingContour(title, testCase, dosave, [x[0],x[-1]], [y[0],y[-1]])
            # con = contour(phi)
            # a = area(con)
            xval, yval = contour(phi)
            a = area(xval, yval)
            print('Area = {0} for {1}'.format(a, title))
            print('Area change = ' + str(100*(initialArea-a)/initialArea) + '%')

        if k%reinitfreq == 0 and k != 0 and doreinit:
            reinitStart = time.time()
            phi = reinit(phi, sc.godunov)
            reinitTime = time.time() - reinitStart
            print('Reinitialization time = {0}'.format(reinitTime))
            totalTime += reinitTime

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
