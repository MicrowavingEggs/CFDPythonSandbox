from Equation import *
from methods import *
from schemes import *
from utils import Sum,Dot

'''
INIT : Ndim, Nx, L, a, dx
'''



class DNSAdvection(Equation):

    def __init__(self,INIT):
        self.INIT = INIT
        try :
            Ndim = INIT["NDIM"]
            ax = INIT["ax"]
            ay = INIT["ay"]
            L = INIT["L"]
            Nx = INIT["Nx"]
            dx = INIT["dx"]
            CFL = INIT["CFL"]
            vmax = self.vmax()
            dt = CFL*dx/vmax
        except:
            raise Exception("One or multiple initField missing.")

        self.tvar = np.zeros([INIT["Nx"]for i in range(Ndim)],dtype = float)
        self.tvar[2:int(0.2*Nx),2:int(0.2*Nx)] = 1
        self.avar = None
        self.isSteady = False

        self.SCHEMES = dict({
                "1X":X1secondOrderCentered,
                "1Y":Y1secondOrderCentered,
                "2X":X2secondOrderCentered,
                "2Y":Y2secondOrderCentered
                })
        self.TIME_METHOD = RK4
        
        @jit(nopython=True)
        def BC(phi):
            for i in range(Nx):
                phi[i][0] = 0
                phi[0][i] = 0
                phi[i][-1] = 0
                phi[-1][i] = 0

        def flux(phi,dx,t=None):
            return -(ax * self.SCHEMES["1X"](phi,dx) + ay * self.SCHEMES["1Y"](phi,dx))
            return -Sum(Dot(ax,self.SCHEMES["1X"](phi,dx)),Dot(ay,self.SCHEMES["1Y"](phi,dx)))

        self.BC = BC
        self.temporalFlux = lambda Phi,t : flux(Phi,dx,t)

    def update(self,tvar,dt,t=None):
        self.TIME_METHOD(tvar,t,self.temporalFlux,dt)
        #tvar += dt*self.temporalFlux(tvar,t)
        self.BC(tvar)


    def vmax(self):
        return 2*np.sqrt(self.INIT["ax"]*self.INIT["ax"] + self.INIT["ay"]*self.INIT["ay"])


    def display(self,tvar,avar=None):
        L = self.INIT["L"]
        print(np.min(tvar))
        print(np.max(tvar))
        plt.imshow(tvar,cmap=plt.get_cmap('coolwarm'),vmin=np.min(tvar), vmax=np.max(tvar), origin='lower',extent=[0,L,0,L])
        plt.show()
        plt.clf()
        if avar != None:
            pass