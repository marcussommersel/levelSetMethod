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

        temp[0, :] = temp[1, :] + (temp[2, :] - temp[1, :])
        temp[-1, :] = temp[-2, :] + (temp[-3, :] - temp[-2, :])

        temp[:, 0] = temp[:, 1] + (temp[:, 2] - temp[:, 1])
        temp[:, -1] = temp[:, -2] + (temp[:, -3] - temp[:, -2])

        phi = temp

    return phi

def area(xval, yval):
    area = np.abs(0.5*np.sum(xval[:-1]*np.diff(yval) - yval[:-1]*np.diff(xval)))
    return area

def contour(fun):
    c = measure.find_contours(fun, 0)
    if len(c) > 1 and testCase == 'zalesak': # Fix for cases where function finds several contours, test if happens for rounded corners.
        c = c[0]
        print("Several contours found! dt might be too high.")
    elif len(c) > 1:
        print("Several contours found! dt might be too high.")
        maxContours = 100
        count = 0
        while count < maxContours:
            c = np.append(c[0], c[1])
            if c.dtype == 'float64':
                break
            count += 1

    con = np.asarray(c, dtype='float64').reshape(-1)
    xval = con[0::2]
    yval = con[1::2]
    return xval/(n-1)*(x[-1] - x[0]), yval/(n-1)*(y[-1] - y[0])

def plottingContour(title='', case='', save=False, limitx=[-1,1], limity=[-1,1]):
    if proj == '3D': # Only for testing purposes
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, phi, 50, cmap='coolwarm')
        plt.show()
        return
    if case == 'vortex': # No need for if
        yoffset = y[0]
    else:
        yoffset = 0
    plt.axis([limitx[0], limitx[1], limity[0], limity[1]])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(initX[0], initX[1], 'b', label='Initial interface')
    xval, yval = contour(phi)
    # plt.plot(xval/(n-1)*(x[-1] - x[0]), yval/(n-1)*(y[-1] - y[0]) + yoffset, 'r', label='Current interface') # Add scaling to contour()?
    plt.plot(xval, yval + yoffset, 'r', label='Current interface') # Add scaling to contour()?
    plt.legend()
    plt.title(title)

    if save:
        a = area(xval, yval)
        if doreinit:
            plt.savefig(('figures/{0}, n = {1}, {2}, A = {3} Reinit iter = {4} at freq = {5}.png'.format(title, n, testCase, a, tmax, reinitfreq)))
        else:
            plt.savefig(('figures/{0}, n = {1}, {2}, A = {3}, no Reinit.png'.format(title, n, testCase, a)))
    plt.close()

if __name__ == '__main__':
    n = 256
    tmax = 10 # number of timesteps in reinitialization
    reinitfreq = 50 # number of iterations between reinitialization
    doreinit = True
    dosave = True
    it = 200001

    CFL = 0.9

    proj = '2D'
    testCase = 'vortex'
    T = 8 # used in vortex-test

    x = np.linspace(0, 1, n)
    if testCase == 'vortex':
        y = np.linspace(0.5, 1.5, n)
    else:
        y = np.linspace(0, 1, n)

    dx = x[1] - x[0]
    dy = y[1] - y[0]


    # print(dx, " ", dy, " ", dx1, " ", dy1)

    u = np.zeros([len(x), len(y)])
    v = np.zeros([len(x), len(y)])
    phi = np.zeros([len(x), len(y)])

    totalTime = 0
    # dt = 0
    # dt =5*10**-5 # From Claudio Walker article [a]
    dt = 0.001 # Slightly below dtmax for vortex with n = 256, CFL = 0.5
    # dt = 0.0025 # Slightly below dtmax for vortex with n = 256, CFL = 0.9
    # dt = 0.0001 # Slightly below dtmax for vortex with n = 512, CFL = 0.9
    # dt = 0.0005 # Slightly below dtmax for vortex with n = 1024, CFL = 0.9
    t = 0

    def uVortex(i,j):
        return -2*((np.sin(np.pi*x[i]))**2)*np.sin(np.pi*y[j])*np.cos(np.pi*y[j])*np.cos(np.pi*t/T) # Claudio Walker
    def vVortex(i,j):
        return -2*np.sin(np.pi*x[i])*np.cos(np.pi*x[i])*((np.cos(np.pi*y[j]))**2)*np.cos(np.pi*t/T) # Claudio Walker
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
        # dt = 0.0025
        # dt = 0.0005
        plotcriteria = T

        initX = [a*np.cos(theta) + 0.5, a*np.sin(theta) + 0.75]
    elif testCase == 'zalesak':

        # Åsmund Ervik:
        cx = 0.5
        cy = 0.5
        # dt = 0.005
        a = 1/3
        # plotcriteria = 628
        plotcriteria = 20

        u = np.fromfunction(uZalesak, (len(x), len(y)), dtype=int)
        v = np.fromfunction(vZalesak, (len(x), len(y)), dtype=int)

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
    elif testCase == 'pospos': # Only used for testing

        dt = 0.25
        plotcriteria = 1

        u[:,:] = -1
        v[:,:] = -1
        a = 0.5
        initX = [a*np.cos(theta), a*np.sin(theta)]

    phi = init(phi, initX)
    xval, yval = contour(phi)
    initialArea = area(xval, yval)
    print("Initial area = {0}".format(initialArea))
    for k in range(it):

        if k%50 == 0:
            print('iteration = {0}, time = {1:.5f}, iteration time = {2:.2f}, t/T = {3:.5f}'.format(k, t, totalTime, t/T))
        if k%reinitfreq == 0 and k != 0 and doreinit:
            reinitStart = time.time()
            phi = reinit(phi, sc.godunov)
            reinitTime = time.time() - reinitStart
            print('Reinitialization time = {0}'.format(reinitTime))
            totalTime += reinitTime
        if k%200 == 0 or round(t/plotcriteria, 4) == 1.00 or round(t/plotcriteria, 4) == 0.25:
            title = 't = {0:.3f}, it = {1}'.format(t, k)
            plottingContour(title, testCase, dosave, [x[0],x[-1]], [y[0],y[-1]])
            xval, yval = contour(phi)
            a = area(xval, yval)
            print('Area = {0} for {1}'.format(a, title))
            print('Area change = ' + str(100*(initialArea-a)/initialArea) + '%')

        startTime = time.time()

        if testCase == 'vortex':
            u = np.fromfunction(uVortex, (len(x), len(y)), dtype=int)
            v = np.fromfunction(vVortex, (len(x), len(y)), dtype=int)

        # CFL condition
        dtmax = CFL/((abs(u)/dx + abs(v)/dy).max()) # From level set book
        if dt > dtmax:
            print('WARNING; dt too high at it = {0}, dt = {1}, dtmax = {2}'.format(k, dt, dtmax))
            break

        t += dt

        phi = sc.TVDRK3(phi, sc.weno, u, v, x, y, dx, dy, dt)

        currentTime = time.time() - startTime
        totalTime += currentTime # plotting not included
