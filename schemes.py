import numpy as np

def TVDRK3(phi, scheme, ax, ay, x, y, dx, dy, dt):

    # first euler step
    phix, phiy = scheme(phi, ax, ay, x, y, dx, dy)
    n1 = phi - dt*(ax*phix + ay*phiy)

    # second euler step
    phix, phiy = scheme(n1, ax, ay, x, y, dx, dy)
    n2 = n1 - dt*(ax*phix + ay*phiy)

    # averaging step
    n1_2 = 3/4*phi + 1/4*n2

    # third euler step
    phix, phiy = scheme(n1_2, ax, ay, x, y, dx, dy)
    n3_2 = n1_2 - dt*(ax*phix + ay*phiy)

    # second averaging step
    temp =  1/3*phi + 2/3*n3_2

    if (scheme) == weno:
        temp = wenoBC(temp)

    return temp

def euler(phi, scheme, ax, ay, x, y, dx, dy, dt):
    phix, phiy = scheme(phi, ax, ay, x, y, dx, dy)
    temp = phi - dt*(ax*phix + ay*phiy)

    if (scheme) == weno:
        temp = wenoBC(temp)

    return temp

def upwind(phi, ax, ay, x, y, dx, dy):
    phix = np.zeros([len(x), len(y)])
    phiy = np.zeros([len(x), len(y)])
    for i in range(2, len(x)-2):
        for j in range(2, len(y)-2):

            if ax[i,j] >= 0:
                phix[i, j] = (phi[i, j] - phi[i - 1, j])/dx
            elif ax[i,j] < 0:
                phix[i, j] = (phi[i + 1, j] - phi[i, j])/dx

            if ay[i,j] >= 0:
                phiy[i, j] = (phi[i, j] - phi[i, j - 1])/dy
            elif ay[i,j] < 0:
                phiy[i, j] = (phi[i, j + 1] - phi[i, j])/dy

    return phix, phiy

def upwind2(phi, ax, ay, x, y, dx,dy):
    phix = np.zeros([len(x), len(y)])
    phiy = np.zeros([len(x), len(y)])
    for i in range(2, len(x)-2):
        for j in range(2, len(y)-2):

            if ax[i, j] >= 0:
                phix[i, j] = (3*phi[i,j] - 4*phi[i-1, j] + phi[i-2, j])/(2*dx)
            elif ax[i, j] < 0:
                phix[i, j] = (-phi[i+2, j] + 4*phi[i+1, j] - 3*phi[i,j])/(2*dx)

            if ay[i, j] >= 0:
                phiy[i, j] = (3*phi[i,j] - 4*phi[i, j-1] + phi[i, j-2])/(2*dy)
            elif ay[i, j] < 0:
                phiy[i, j] = (-phi[i, j+2] + 4*phi[i, j+1] - 3*phi[i,j])/(2*dy)

    return phix, phiy

def central(phi, ax, ay, x, y, dx, dy):
    phix = np.zeros([len(x), len(y)])
    phiy = np.zeros([len(x), len(y)])
    for i in range(2, len(x)-2):
        for j in range(2, len(y)-2):

            phix[i, j] = (phi[i + 1, j] - phi[i - 1, j])/(2*dx)
            phiy[i, j] = (phi[i, j + 1] - phi[i, j - 1])/(2*dy)

    return phix, phiy

def godunov(phi, ax, ay, x, y, dx, dy):
    phix = np.zeros([len(x), len(y)])
    phiy = np.zeros([len(x), len(y)])
    for i in range(2, len(x)-2):
        for j in range(2, len(y)-2):
            phix_m = (phi[i,j] - phi[i-1,j])/dx
            phix_p = (phi[i+1,j] - phi[i,j])/dx
            
            phiy_m = (phi[i,j] - phi[i,j-1])/dy
            phiy_p = (phi[i,j+1] - phi[i,j])/dy

            if ax[i,j] >= 0:
                phix[i,j] = np.sqrt(max(max(phix_m, 0)**2, min(phix_p, 0)**2))
            elif ax[i,j] < 0:
                phix[i,j] = np.sqrt(max(min(phix_m, 0)**2, max(phix_p, 0)**2))

            if ay[i,j] >= 0:
                phiy[i,j] = np.sqrt(max(max(phiy_m, 0)**2, min(phiy_p, 0)**2))
            elif ay[i,j] < 0:
                phiy[i,j] = np.sqrt(max(min(phiy_m, 0)**2, max(phiy_p, 0)**2))
    
    return phix, phiy

def weno(phi, ax, ay, x, y, dx, dy):
    phix = np.zeros([len(x), len(y)])
    phiy = np.zeros([len(x), len(y)])
    for i in range(2, len(x)-3):
        for j in range(2, len(y)-3):

            # WENO
            if ax[i,j] >= 0:
                v1 = (phi[i-2,j] - phi[i-3,j])/dx
                v2 = (phi[i-1,j] - phi[i-2,j])/dx
                v3 = (phi[i,j] - phi[i-1,j])/dx
                v4 = (phi[i+1,j] - phi[i,j])/dx
                v5 = (phi[i+2,j] - phi[i+1,j])/dx
            elif ax[i,j] < 0:
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

            if ay[i,j] >= 0:
                v1 = (phi[i,j-2] - phi[i,j-3])/dy
                v2 = (phi[i,j-1] - phi[i,j-2])/dy
                v3 = (phi[i,j] - phi[i,j-1])/dy
                v4 = (phi[i,j+1] - phi[i,j])/dy
                v5 = (phi[i,j+2] - phi[i,j+1])/dy
            elif ay[i,j] < 0:
                v1 = (phi[i,j-1] - phi[i,j-2])/dy
                v2 = (phi[i,j] - phi[i,j-1])/dy
                v3 = (phi[i,j+1] - phi[i,j])/dy
                v4 = (phi[i,j+2] - phi[i,j+1])/dy
                v5 = (phi[i,j+3] - phi[i,j+2])/dy

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
