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

        temp[0, :] = temp[1, :] + (temp[2, :] - temp[1, :])
        temp[-1, :] = temp[-2, :] + (temp[-3, :] - temp[-2, :])

        temp[:, 0] = temp[:, 1] + (temp[:, 2] - temp[:, 1])
        temp[:, -1] = temp[:, -2] + (temp[:, -3] - temp[:, -2])

        phi = temp

    return phi

def interfaceError(computed, expected):
    error  = 0
    L = 0
    xval, yval = contour(expected)
    for k in range(len(xval[0])-1):
        L += np.sqrt((xval[0][k+1] - xval[0][k])**2 + (yval[0][k+1] - yval[0][k])**2)
    for i in range(len(computed[:,0])):
        for j in range(len(computed[0,:])):
            error += abs((expected[i,j]<0)*1 - (computed[i,j]<0)*1)*dx*dy
    return error/L

def mt(vec):
    vec[vec >= 0] = 0
    vec[vec < 0] = 1
    M_t = np.sum(np.abs(vec)*dx*dy)
    return M_t

def massError(vec, tend):
    return np.sum(vec)/tend

def area(xval, yval):
    area = 0
    for i in range(len(xval)):
        area += np.abs(0.5*np.sum(xval[i][:-1]*np.diff(yval[i]) - yval[i][:-1]*np.diff(xval[i])))
    return area

def contour(fun):
    c = measure.find_contours(fun, 0)
    con = []
    xval = []
    yval = []
    for i in range(len(c)):
        con.append(np.asarray(c[i], dtype='float64').reshape(-1))
        xval.append(con[i][0::2])
        yval.append(con[i][1::2])
    xval = np.asarray(xval, dtype='object')
    yval = np.asarray(yval, dtype='object')
    return xval/(n-1)*(x[-1] - x[0]) + x[0], yval/(n-1)*(y[-1] - y[0]) +  y[0]

def plottingContour(xcompare, ycompare, title='', case='', save=False, limitx=[-1,1], limity=[-1,1]):
    if proj == '3D': # Only for testing purposes
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, phi, 50, cmap='coolwarm')
        plt.show()
        return
    plt.axis([limitx[0], limitx[1], limity[0], limity[1]])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x', fontsize=14, style='italic')
    plt.ylabel('y', fontsize=14, style='italic')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(xcompare, ycompare, 'b', label='Initial interface')
    xval, yval = contour(phi)
    for i in range(len(xval)):
        plt.plot(xval[i], yval[i], 'r', label='Current interface'*(i==0))
    if save:
        a = area(xval, yval)
        if doreinit:
            plt.savefig(savePath + '{0}, {1}, n = {2}, dt = {3}, Reinit iter = {4} at freq = {5}, A = {6}.png'.format(testCase, title, n, dt, tmax, reinitfreq, a), dpi=900)
            np.savetxt(savePath + '{0}, {1}, n = {2}, dt = {3}, Reinit iter = {4} at freq = {5}, A = {6}.csv'.format(testCase, title, n, dt, tmax, reinitfreq, a), phi, delimiter=',')
        else:
            plt.savefig(savePath + '{0}, {1}, n = {2}, dt = {3}, no Reinit, A = {4}.png'.format(testCase, title, n, dt, a), dpi=900)
            np.savetxt(savePath + '{0}, {1}, n = {2}, dt = {3}, no Reinit, A = {4}.csv'.format(testCase, title, n, dt, a), phi, delimiter=',')
    plt.show()

if __name__ == '__main__':

    startTime = time.time()

# Variables
######################################################################################
    n = 128
    tmax = 10
    reinitfreq = 50
    printfreq = 500
    plotfreq = 1000
    doreinit = True
    dosave = False

    CFL = 0.5
    it = 20000 # Maximum number of iterations
    proj = '2D'
    testCase = 'vortex'
    T = 2 # End time
######################################################################################

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    u = np.zeros([len(x), len(y)])
    v = np.zeros([len(x), len(y)])
    phi = np.zeros([len(x), len(y)])
    theta = np.linspace(0, 2*np.pi, n)
    dt = 0
    t = 0

    def uVortex(i,j):
        return -2*(np.sin(np.pi*x[i]))**2*np.sin(np.pi*y[j])*np.cos(np.pi*y[j])*np.cos(np.pi*t/T)
    def vVortex(i,j):
        return 2*(np.sin(np.pi*y[j]))**2*np.sin(np.pi*x[i])*np.cos(np.pi*x[i])*np.cos(np.pi*t/T)
    def uZalesak(i,j):
        return np.pi/10*(0.5 - y[j])
    def vZalesak(i,j):
        return np.pi/10*(x[i] - 0.5)

    if testCase == 'vortex':

        a = 0.15
        cx = 0.5
        cy = 0.75
        initX = [a*np.cos(theta) + cx, a*np.sin(theta) + cy]
        # it = int(T/dt + 1) # If a constant dt is chosen
    elif testCase == 'zalesak':

        cx = 0.5
        cy = 0.5
        a = 1/3
        T = 20
        # it = int(20/dt + 1) # If a constant dt is chosen

        u = np.fromfunction(uZalesak, (len(x), len(y)), dtype=int)
        v = np.fromfunction(vZalesak, (len(x), len(y)), dtype=int)

        width = a/3
        length = width*5
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
        T = 1
        cx = 0.5
        cy = 0.5
        u[:,:] = -1
        v[:,:] = -1
        a = 0.5
        initX = [a*np.cos(theta) + cx, a*np.sin(theta) + cy]

    if dosave:
        savePath = 'figures/{0}/n = {1}/'.format(testCase, n)
    else:
        savePath = 'figures/'
    
    phi = init(phi, initX)
    mtVec = []
    phiInit = np.copy(phi)
    phiInitX, phiInitY = contour(phiInit)
    mt_0 = mt(np.copy(phiInit))

    xval, yval = contour(phi)
    initialArea = area(xval, yval)
    print('Initial area = {0}'.format(initialArea))
    title = 't = {0:.3f}, it = 0'.format(t)
    plottingContour(phiInitX, phiInitY, title, testCase, dosave, [x[0],x[-1]], [y[0],y[-1]])

    for k in range(1, it):

        if testCase == 'vortex': # velocity of vortex test varies with t
            u = np.fromfunction(uVortex, (len(x), len(y)), dtype=int)
            v = np.fromfunction(vVortex, (len(x), len(y)), dtype=int)

        # CFL condition
        dt = CFL/((abs(u)/dx + abs(v)/dy).max()) # From level set book; Osher and Fedkiw (2003).

        if t + dt > T:
            dt = T - t
        
        t += dt
        
        phi = sc.TVDRK3(phi, sc.weno, u, v, x, y, dx, dy, dt)
        mtVec = np.append(mtVec, np.abs(mt(np.copy(phi)) - mt_0)*dt)

        if k%reinitfreq == 0 and k != 0 and doreinit: # Reinitialization
            reinitStart = time.time()
            phi = reinit(phi, sc.godunov)
            reinitTime = time.time() - reinitStart
            print('Reinitialization time = {0}'.format(reinitTime))
        if k%plotfreq == 0 or round(t/T, 4) == 1.000: # Plotting
            title = 't = {0:.3f}, it = {1}'.format(t, k)
            xval, yval = contour(phi)
            plottingContour(phiInitX[0], phiInitY[0], title, testCase, dosave, [x[0],x[-1]], [y[0],y[-1]])
            a = area(xval, yval)
            print('Area = {0} for {1}'.format(a, title))
            print('Area change = ' + str(100*(initialArea-a)/initialArea) + '%')
        if k%printfreq == 0: # Printing
            print('iteration = {0}, time = {1:.5f}, iteration time = {2:.2f}'.format(k, t, time.time() - startTime) + (testCase=='vortex')*', t/T = {0:.5f}'.format(t/T))

        if t == T:
            break

    plottingContour(phiInitX[0], phiInitY[0], title, testCase, dosave, [x[0],x[-1]], [y[0],y[-1]])
    
    if dosave:
        f = open(savePath + 'log.txt', 'w')
        f.write('Total time = {0:.4f}\n'.format(time.time() - startTime))
        f.write('Interface error = {0}\n'.format(interfaceError(phi, phiInit)))
        f.write('Mass error = {0}\n'.format(massError(mtVec, t)))
        f.write('Area = {0} for {1}\n'.format(a, title))
        f.write('Area change = {0} %\n'.format(100*(initialArea-a)/initialArea))
        f.close()
    else:
        print('')
        print('Total time = {0:.4f}'.format(time.time() - startTime))
        print('Interface error = {0}'.format(interfaceError(phi, phiInit)))
        print('Mass error = {0}'.format(massError(mtVec, t)))
        print('Area = {0} for {1}'.format(a, title))
        print('Area change = {0} %'.format(100*(initialArea-a)/initialArea))
