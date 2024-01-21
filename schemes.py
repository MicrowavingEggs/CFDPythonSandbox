import numpy as np
from numba import jit

@jit(nopython=True,parallel=True)
def X1secondOrderCentered(phi,h): #computes 2nd-order Centered Difference for f' at (x,y), for t such that phi = PHI(t,x,y)
    res = np.zeros((len(phi),len(phi[0])),dtype = float)
    for j in range(len(phi)):
        for i in range(len(phi[0])):
            if i == len(phi[0])-1:
                res[j][i] = 0
            elif i == 0:
                res[j][i] = 0
            else:
                res[j][i] = (phi[j][i+1]-phi[j][i-1])/(2*h)
    return res

@jit(nopython=True,parallel=True)
def Y1secondOrderCentered(phi,h): #computes 2nd-order Centered Difference for f' at (x,y), for t such that phi = PHI(t,x,y)
    res = np.zeros((len(phi),len(phi[0])),dtype = float)
    for j in range(len(phi)):
        for i in range(len(phi[0])):
            if j == len(phi)-1:
                res[j][i] = 0
            elif j == 0:
                res[j][i] = 0
            else:
                res[j][i] = (phi[j+1][i]-phi[j-1][i])/(2*h)
    return res

@jit(nopython=True,parallel=True)
def X2secondOrderCentered(phi,h):
    res = np.zeros((len(phi),len(phi[0])),dtype = float)
    for j in range(len(phi)):
        for i in range(len(phi[0])):
            if  i == len(phi)-1:
                res[j][i] = 0
            elif i == 0:
                res[j][i] = 0
            else:
                res[j][i] = (phi[j][i+1]-2*phi[j][i]+phi[j][i-1])/(h*h)
    return res


@jit(nopython=True,parallel=True)
def Y2secondOrderCentered(phi,h):
    res = np.zeros((len(phi),len(phi[0])),dtype = float)
    for j in range(len(phi)):
        for i in range(len(phi[0])):
            if j == len(phi)-1:
                res[j][i] = 0
            elif j == 0:
                res[j][i] = 0
            else:
                res[j][i] = (phi[j+1][i]-2*phi[j][i]+phi[j-1][i])/(h*h)
    return res


### OPERATORS

def LaplacianX(phi,SCHEMES,h):
    return SCHEMES["2X"](phi,h) + SCHEMES["2Y"](phi,h)

def LaplacianY(phi,SCHEMES,h):
    return SCHEMES["2X"](phi,h) + SCHEMES["2Y"](phi,h)

def Laplacian2D(phi,SCHEMES,Nx,Ny,L=5):
    res = np.zeros([len(phi),len(phi[0]),2],dtype = float)
    hx = L/(Nx-1)
    hy = L/(Ny-1)

    res[:,:,0] = LaplacianX(phi[:,:,0],SCHEMES,hx)
    res[:,:,1] = LaplacianY(phi[:,:,1],SCHEMES,hy)
    return res

def Laplacian1D(phi,SCHEMES,Nx,Ny,L=5):
    return SCHEMES["2X"](phi,L/(Nx-1)) + SCHEMES["2Y"](phi,L/(Ny-1))