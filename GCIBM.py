import schemes as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
# import matplotlib
import time
from skimage import measure
import tikzplotlib

def init(phi, init):
    p = path.Path(np.transpose(init))
    for i in range(len(x)):
        for j in range(len(y)):
            if p.contains_points([(x[i], y[j])]):
                phi[i,j] = - min(np.sqrt((x[i] - init[0])**2 + (y[j] - init[1])**2))
            else:
                phi[i,j] = min(np.sqrt((x[i] - init[0])**2 + (y[j] - init[1])**2))
    return phi

def cent(phi, x, y, dx, dy):
    phix = np.zeros([len(x), len(y)])
    phiy = np.zeros([len(x), len(y)])
    for i in range(2, len(x)-2):
        for j in range(2, len(y)-2):

            phix[i, j] = (phi[i + 1, j] - phi[i - 1, j])/(2*dx)
            phiy[i, j] = (phi[i, j + 1] - phi[i, j - 1])/(2*dy)
    return phix, phiy

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

def plottingContour(title='', case='', save=False, limitx=[0,1], limity=[0,1]):
    if proj == '3D': # Only for testing purposes
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, phi, 50, cmap='coolwarm')
        plt.show()
        return
    plt.axis([limitx[0], limitx[1], limity[0], limity[1]])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x', fontsize=12, style='italic')
    plt.ylabel('y', fontsize=12, style='italic')
    plt.plot(initX[0], initX[1], 'b', label='Initial interface')
    xval, yval = contour(phi)
    for i in range(len(xval)):
        plt.plot(xval[i], yval[i], 'r', label='Current interface'*(i==0))
    plt.legend(loc='upper right', fontsize=11)
    plt.grid()
    plt.show()

if __name__ == '__main__':

    # matplotlib.font_manager.findfont('NimbusRomNo9L', 'ttf', None, True, True)
    startTime = time.time()
#######################################################################################
    n = 401
    tmax = 10 # number of timesteps in reinitialization, 10 best for zalesak
    reinitfreq = 500 # number of iterations between reinitialization, 500 best for zalesak, 250 for vortex at T=2, 500 at T = 8
    printfreq = 500
    plotfreq = 2000 # 4000 = end for vortex, T = 2, 8000 = end for zalesak,
    dosave = False

    proj = '2D'
    testCase = 'pospos'
#######################################################################################

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    x = np.linspace(0, 1, n)
    if testCase == 'vortex':
        y = np.linspace(0.5, 1.5, n)
    else:
        y = np.linspace(0, 1, n)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    u = np.zeros([len(x), len(y)])
    v = np.zeros([len(x), len(y)])
    phi = np.zeros([len(x), len(y)])
    theta = np.linspace(0, 2*np.pi, n)

    if testCase == 'vortex':

        # Claudio Walker:
        a = 0.15
        cx = 0.5
        cy = 0.75

        initX = [a*np.cos(theta) + 0.5, a*np.sin(theta) + 0.75]
    elif testCase == 'zalesak':

        # Åsmund Ervik:
        cx = 0.5
        cy = 0.5
        # dt = 0.005
        a = 1/3

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
        a = 0.4
        cx = 0.5
        cy = 0.5
        initX = [a*np.cos(theta) + cx, a*np.sin(theta) + cy]

    phi = init(phi, initX)
    xval, yval = contour(phi)
    initialArea = area(xval, yval)
    # plottingContour(testCase, dosave, [x[0],x[-1]], [y[0],y[-1]])
    xval, yval = contour(phi)

    #plt.axis([0, 1, 0, 1])
    plt.axis([-0.1, 1.1, 0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x', fontsize=14, style='italic')
    plt.ylabel('y', fontsize=14, style='italic')
    plt.plot(initX[0], initX[1], 'b', label='Interface')
    xval, yval = contour(phi)
    # for i in range(len(xval)):
    #     plt.plot(xval[i], yval[i], 'r', label='Current interface'*(i==0))
    # plt.legend(loc='upper right', fontsize=11)
#######################################################################################
    phix, phiy = cent(phi, x, y, dx, dy)
    grad = np.sqrt(phix**2 + phiy**2)
    gradx = phix
    grady = phiy

    # alpha = np.arccos(phix/phi)
    
    # nn = grad/abs(grad)
    nx = gradx#/abs(gradx)
    ny = grady#/abs(grady)



    x1 = n*2/10
    y1 = n*6/10

    print(nx[int(x1)+1, int(y1)+1])
    print(ny[int(x1)+1, int(y1)+1])

    ## only nn
    # bix = x[int(x1)] - phi[int(x1), int(y1)]*nn[int(x1)+1, int(y1)+1]
    # ipx = x[int(x1)] - 2*phi[int(x1), int(y1)]*nn[int(x1)+1, int(y1)+1]

    # biy = y[int(y1)] - phi[int(x1), int(y1)]*nn[int(x1)+1, int(y1)+1]
    # ipy = y[int(y1)] - 2*phi[int(x1), int(y1)]*nn[int(x1)+1, int(y1)+1]

    plt.plot(x[int(x1)], y[int(y1)], 'ro', label='Ghost point', ms=8)
    # plt.plot(bix, biy, 'go', label='Boundary intercept nn')
    # plt.plot(ipx, ipy, 'co', label='Image point nn')

    ## nx and ny
    bix = x[int(x1)] - phi[int(x1), int(y1)]*nx[int(x1)+1, int(y1)+1]
    ipx = x[int(x1)] - 2*phi[int(x1), int(y1)]*nx[int(x1)+1, int(y1)+1]
    
    biy = y[int(y1)] - phi[int(x1), int(y1)]*ny[int(x1)+1, int(y1)+1]
    ipy = y[int(y1)] - 2*phi[int(x1), int(y1)]*ny[int(x1)+1, int(y1)+1]


    fpx1 = 0
    fpx2 = 0.2
    fpy1 = 0.6
    fpy2 = 0.8

    plt.plot(fpx1, fpy1, 'kx', label='Fluid point', ms=8)
    plt.plot(bix, biy, 'gx', label='Boundary intercept', ms=8)
    plt.plot(ipx, ipy, 'mx', label='Image point', ms=8)
    plt.plot(fpx1, fpy2, 'kx', ms=8)
    plt.plot(fpx2, fpy2, 'kx', ms=8)
    plt.plot()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True)
    plt.savefig('figures/GCIBM.png', dpi=900)

    plt.show()
