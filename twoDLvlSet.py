import numpy as np
import matplotlib.pyplot as plt

def init(phi, init = False):
    if init:
        for i in range(len(phi[:, 0])):
            for j in range(len(phi[0,:])):
                phi[i,j] = min(np.sqrt(x[i]**2 + y[j]**2) - np.sqrt(init[0]**2 + init[1]**2))
    else:
        print("b")
        init = [[], []]
        for i in range(len(x)):
            for j in range(len(y)):
                if round(phi[i,j], 1) == 0:
                    init[0].append(x[i])
                    init[1].append(y[j])
        for i in range(len(phi[:, 0])):
            for j in range(len(phi[0,:])):
                phi[i,j] = min(np.sqrt(x[i]**2 + y[j]**2) - np.sqrt(np.power(init[0], 2) + np.power(init[1], 2)))
    return phi

def reinit(phi):
    temp = phi
    for k in range(tmax):
        for i in range(2, len(x)-2):
            for j in range(2, len(y)-2):
                
                # WENO
                # omega1 = 0.1
                # omega2 = 0.6
                # omega3 = 0.3

                v1 = (phi[i-2,j] - phi[i-3,j])/dx
                v2 = (phi[i-1,j] - phi[i-2,j])/dx
                v3 = (phi[i,j] - phi[i-1,j])/dx
                v4 = (phi[i+1,j] - phi[i,j])/dx
                v5 = (phi[i+2,j] - phi[i+1,j])/dx

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

                uphix = u[i,j]*(omega1*phix1 + omega2*phix2 + omega3*phix3)

                v1 = (phi[i,j-2] - phi[i,j-3])/dy
                v2 = (phi[i,j-1] - phi[i,j-2])/dy
                v3 = (phi[i,j] - phi[i,j-1])/dy
                v4 = (phi[i,j+1] - phi[i,j])/dy
                v5 = (phi[i,j+2] - phi[i,j+1])/dy

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

                vphiy = v[i,j]*(omega1*phiy1 + omega2*phiy2 + omega3*phiy3)

                # 2nd upwind
                # uphix = (max(u[i,j], 0)*(3*phi[i,j] - 4*phi[i-1, j] + phi[i-2, j])/(2*dx) 
                #     + min(u[i,j], 0)*(-phi[i+2, j] + 4*phi[i+1, j] - 3*phi[i,j])/(2*dx))

                # vphiy = (max(v[i,j], 0)*(3*phi[i,j] - 4*phi[i, j-1] + phi[i, j-2])/(2*dy) 
                #     + min(v[i,j], 0)*(-phi[i, j+2] + 4*phi[i, j+1] - 3*phi[i,j])/(2*dy))

                temp[i,j] =  phi[i,j] - dt*S0[i,j]*(np.sqrt(uphix**2 + vphiy**2) - 1)

                # Euler and central
                # phi_norm = np.sqrt(((temp[i+1,j] - temp[i-1,j])/(2*dx))**2 + ((temp[i,j+1] - temp[i,j-1])/(2*dy))**2)
                # temp[i,j] = phi[i,j] - dt*S0[i,j]*(phi_norm - 1)

        temp[2, :] = temp[3, :] - (temp[4, :]- temp[3, :])
        temp[1, :] = temp[2, :] - (temp[3, :]- temp[2, :])
        temp[0, :] = temp[1, :] - (temp[2, :]- temp[1, :])
        temp[-3, :] = temp[-4, :] - (temp[-5, :]- temp[-4, :])
        temp[-2, :] = temp[-3, :] - (temp[-4, :]- temp[-3, :])
        temp[-1, :] = temp[-2, :] - (temp[-3, :]- temp[-2, :])

        temp[:, 2] = temp[:, 3] - (temp[:, 4]- temp[:, 3])
        temp[:, 1] = temp[:, 2] - (temp[:, 3]- temp[:, 2])
        temp[:, 0] = temp[:, 1] - (temp[:, 2]- temp[:, 1])
        temp[:, -3] = temp[:, -4] - (temp[:, -5]- temp[:, -4])
        temp[:, -2] = temp[:, -3] - (temp[:, -4]- temp[:, -3])
        temp[:, -1] = temp[:, -2] - (temp[:, -3]- temp[:, -2])

        # Old BC fdkjsalø
        # temp[0,:] = phi[0,:] + (u[0] + epsilon)/abs(u[0] + epsilon)*(phi[0,:] - phi[1,:])
        # temp[-1,:] = phi[-1,:] + (u[0] + epsilon)/abs(u[-1] + epsilon)*(phi[-2,:] - phi[-1,:])
        # temp[:,0] = phi[:,0] + (v[0] + epsilon)/abs(v[0] + epsilon)*(phi[:,0] - phi[:,1])
        # temp[:,-1] = phi[:,-1] + (v[0] + epsilon)/abs(v[-1] + epsilon)*(phi[:,-2] - phi[:,-1])
        phi = temp
        # plottingContour()
    return phi

def plottingContour(title = ''):
    m = 1
    if proj == '2D':
        plt.plot(initX[0], initX[1], 'r')
        plt.contourf(x[m:-m], y[m:-m], phi[m:-m,m:-m])
        plt.colorbar()
    elif proj == '3D':
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, phi, 50, cmap='coolwarm')
    plt.title(title)
    plt.show()

n = 100
tmax = 10 # used in reinitialization
it = 100
proj = "2D"
epsilon = 10e-6

dx = 1/n*0.001
dy = dx

x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)

u = np.zeros([len(x), len(y)])
v = np.zeros([len(x), len(y)])
u[:,:] = 1
v[:,:] = 1

dt = (dx + dy)/(u + v).max() # CFL condition

phi = np.zeros([len(x), len(y)])

a = 0.5
theta = np.linspace(0, 2*np.pi, n)
initX = [a*np.cos(theta), a*np.sin(theta)]

phi = init(phi, initX)
phi0 = phi
S0 = phi0/(np.sqrt(phi0**2 + dx**2))

# Godunov, boken til sethien.

for k in range(it):
    if k%2 == 0 and k != 0:
        phi = reinit(phi)
    plottingContour(k)
    temp = np.zeros_like(phi)

    n1 = np.zeros_like(phi)
    n2 = np.zeros_like(phi)
    n1_2 = np.zeros_like(phi)
    n3_2 = np.zeros_like(phi)
    
    for i in range(2,len(x)-2):
        for j in range(2,len(y)-2):
            # temp[i,j] = phi[i, j] - dt/2*(u[i,j]*(phi[i+1, j] - phi[i-1, j])/dx 
            #     + v[i,j]*(phi[i, j + 1] - phi[i, j - 1])/dy) # central and 1st euler

            # RK with central differencing
            # n1[i,j] = phi[i,j] - dt/2*(u[i,j]*(phi[i+1, j] - phi[i-1, j])/dx 
            #     + v[i,j]*(phi[i, j + 1] - phi[i, j - 1])/dy) # central
            
            # n2[i,j] = n1[i,j] - dt/2*(u[i,j]*(n1[i + 1, j] - n1[i-1, j])/dx 
            #     + v[i,j]*(n1[i, j + 1] - n1[i, j - 1])/dy)
            
            # n1_2[i,j] = 3/4*phi[i,j] + 1/4*n2[i,j]

            # n3_2[i,j] = n1_2[i,j] - dt/2*(u[i,j]*(n1_2[i + 1, j] - n1_2[i-1, j])/dx 
            #     + v[i,j]*(n1_2[i, j + 1] - n1_2[i, j - 1])/dy)
            
            # temp[i,j] = 1/3*phi[i,j] + 2/3*n3_2[i,j]

            # omega1 = 0.1
            # omega2 = 0.6
            # omega3 = 0.3

            v1 = (phi[i-2,j] - phi[i-3,j])/dx
            v2 = (phi[i-1,j] - phi[i-2,j])/dx
            v3 = (phi[i,j] - phi[i-1,j])/dx
            v4 = (phi[i+1,j] - phi[i,j])/dx
            v5 = (phi[i+2,j] - phi[i+1,j])/dx

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

            phix = omega1*phix1 + omega2*phix2 + omega3*phix3

            v1 = (phi[i,j-2] - phi[i,j-3])/dy
            v2 = (phi[i,j-1] - phi[i,j-2])/dy
            v3 = (phi[i,j] - phi[i,j-1])/dy
            v4 = (phi[i,j+1] - phi[i,j])/dy
            v5 = (phi[i,j+2] - phi[i,j+1])/dy

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

            phiy = omega1*phiy1 + omega2*phiy2 + omega3*phiy3

            temp[i,j] = phi[i,j] - dt*(u[i,j]*phix + v[i,j]*phiy)

    # BC HJ WENO + Euler
    temp[2, :] = temp[3, :] - (temp[4, :]- temp[3, :])
    temp[1, :] = temp[2, :] - (temp[3, :]- temp[2, :])
    temp[0, :] = temp[1, :] - (temp[2, :]- temp[1, :])
    temp[-2, :] = temp[-3, :] - (temp[-4, :]- temp[-3, :])
    temp[-1, :] = temp[-2, :] - (temp[-3, :]- temp[-2, :])

    temp[:, 2] = temp[:, 3] - (temp[:, 4]- temp[:, 3])
    temp[:, 1] = temp[:, 2] - (temp[:, 3]- temp[:, 2])
    temp[:, 0] = temp[:, 1] - (temp[:, 2]- temp[:, 1])
    temp[:, -2] = temp[:, -3] - (temp[:, -4]- temp[:, -3])
    temp[:, -1] = temp[:, -2] - (temp[:, -3]- temp[:, -2])
    phi = temp
    

    # BC central + euler        
    # temp[0,:] = phi[0,:] + (u[0] + epsilon)/abs(u[0] + epsilon)*(phi[0,:] - phi[1,:])
    # temp[-1,:] = phi[-1,:] + (u[0] + epsilon)/abs(u[-1] + epsilon)*(phi[-2,:] - phi[-1,:])
    # temp[:,0] = phi[:,0] + (v[0] + epsilon)/abs(v[0] + epsilon)*(phi[:,0] - phi[:,1])
    # temp[:,-1] = phi[:,-1] + (v[0] + epsilon)/abs(v[-1] + epsilon)*(phi[:,-2] - phi[:,-1])
    # phi = temp

if __name__ == "__main__":
    print(dt)