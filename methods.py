import numpy as np
from numba import jit
from utils import Sum,Dot
import timeit
from schemes import *
###MISC

def clockFunc(f): #returns the runtime duration of f. f = lambda func.
    start = timeit.default_timer()
    f()
    stop = timeit.default_timer()
    return stop - start

def MatrixToVec(A): #v is a matrix
    res = np.zeros(len(A)*len(A))
    for i in range(len(A)):
        for j in range(len(A[i])):
            res[j][i] = A[i][j]
    return res

def sourceToVec(source,N,L): #source is a function
    res = np.zeros(N*N)
    for i in range(N):
        for j in range(N):
            res[i + j*N] = source(i*L/(N-1),j*L/(N-1))
    return res

def sourceToMat(source,N,L): #source is a function
    res = np.zeros([N,N],dtype = float)
    h = (L/(N-1))
    for i in range(N):
        for j in range(N):
            res[j][i] = (h*h)*source(i*h,j*h)
    return res

def VecTomatrix(v):
    n = int(np.sqrt(len(v)))
    return v.reshape((n,n))

def isInversible(A): #Useless : secure check
    return np.matrix_rank(A) == (len(A))

def L2Norm(A,spaceStep):#A is a matrix
    res = 0
    for j in range(1,len(A)-1):
        for i in range(1,len(A[0])-1):
            res += spaceStep*spaceStep*(A[j][i] * A[j][i])
    return np.sqrt(res)

def residual(Phi,sourceMat,SCHEMES,dx): #computes r = ||Laplacian(P) - source||_2
    return L2Norm(Laplacian1D(Phi,SCHEMES,len(Phi),len(Phi[0])) - sourceMat,dx)

###METHODS

### TEMPORAL
def EE(Phi,t,f,ht):
    Phi += ht*f(Phi,t)

def RK2(Phi,t,f,ht): #f=GeometricalScheme-like function.
    k1 = f(Phi,t)
    k2 = f(Phi + 0.5*ht*k1,(t+0.5)*ht)
    Phi += ht*k2

def RK3(Phi,t,f,ht): #KUTTA3 SCHEME
    k1 = f(Phi,t)
    k2 = f(Phi + 0.5*ht*k1,(t+0.5)*ht)
    k3 = f(Phi - ht*k1 +2*ht*k2,(t+1)*ht)
    Phi += ht*k1/6 + 2*ht*k2/3 + ht*k3/6

def RK4(Phi,t,f,ht): #CLASSICAL SCHEME
    k1 = f(Phi,t)
    k2 = f(Phi + 0.5*ht*k1,(t+0.5)*ht)
    k3 = f(Phi + 0.5*ht*k2,(t+0.5)*ht)
    k4 = f(Phi + ht*k3,(t+1)*ht)
    Phi += (ht*(k1 + 2*k2 + 2*k3 + k4)/6)

def LeapFrog(Phi,t,f,ht): #LeapFrog #TODO : NOT WORKING
    if t == 1:
        return Phi[t-1] + ht*f(Phi[t-1],t)
    else:
        return Phi[t-2] + 2*ht*f(Phi[t-1],t)


### STEADY
def GaussSeidelSimple(Phi,source,N,L=0.3): #clim is a mapping func
    b = sourceToMat(source,N,L)#sourceToVec(source,N,L)
    Phinew = Phi #np.zeros([N,N],dtype = float) #np.zeros(N*N)
    h = L/(N-1)

    epsilon = 0.0001 #Threshold
    r0 = residual(Phi,b,h)

    while ((r := residual(Phi,b,h)) > epsilon*r0):
        print(r)
        for i in range(1,N-1):
            for j in range(1,N-1):
                Phinew[j][i] = 0.25*(Phi[j][i+1]+Phinew[j][i-1]+Phi[j+1][i]+Phinew[j-1][i]) - 0.25*b[j][i]
        Phi = Phinew
    return Phi

@jit(nopython=True)
def SORLoop(Phinew,Phi,omega,source,Nx,dx):
    for i in range(1,Nx-1):
        for j in range(1,Nx-1):
            Phinew[j][i] = (1-omega)*Phi[j][i] + omega*(0.25*(Phi[j][i+1]+Phinew[j][i-1]+Phi[j+1][i]+Phinew[j-1][i]) - 0.25*dx*dx*source[j][i])

def SOR(Phi,source,BC,dx,epsilon = 0.001):
    Phinew = Phi #np.zeros([N,N],dtype = float) #np.zeros(N*N)
    SCHEMES = dict({
                "1X":X1secondOrderCentered,
                "1Y":Y1secondOrderCentered,
                "2X":X2secondOrderCentered,
                "2Y":Y2secondOrderCentered
                })
    Nx = len(Phi)
    r0 = residual(np.zeros([Nx,Nx],dtype = float),source,SCHEMES,dx)    
    omega = 2/(1+np.sin(np.pi/Nx))


    while ((r := residual(Phi,source,SCHEMES,dx)) > epsilon*r0):
        SORLoop(Phinew,Phi,omega,source,Nx,dx)
        BC(Phinew)
        Phi = Phinew
    return Phi