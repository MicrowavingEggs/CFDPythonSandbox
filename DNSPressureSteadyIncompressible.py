from Equation import *
from methods import SOR

'''
INIT : Nx, L, Uc, rho, dx,epsilon
'''



class DNSPressureSteadyIncompressible(Equation):

    def __init__(self,INIT):
        self.tvar = np.zeros([INIT["Nx"],INIT["Nx"]],dtype = float)
        self.avar = None
        self.isSteady = True
        self.INIT = INIT
        try :
            Uc = INIT["Uc"]
            rho = INIT["rho"]
            L = INIT["L"]
            Nx = INIT["Nx"]
        except:
            raise Exception("One or multiple initField missing.")
        pi = np.pi
        
        @jit(nopython=True,parallel=True)
        def BC(P):
            for i in range(Nx):
                P[i][0] = 0.25*rho*Uc*Uc*(1-np.cos(2*pi*(i/(Nx-1))))
                P[0][i] = 0.25*rho*Uc*Uc*(1-np.cos(2*pi*(i/(Nx-1))))
                P[i][-1] = P[i][-2]
                P[-1][i] = P[-2][i]
        
        self.source = np.array([[rho*((pi*Uc/L)**2)*(np.cos(2*pi*i/(Nx-1)) + np.cos(2*pi*j/(Nx-1)))for i in range(Nx)]for j in range(Nx)])
        self.BC = BC

    def update(self,tvar,t=None):
        if "epsilon" in self.INIT.keys():
            eps = self.INIT["epsilon"]
        else:
            eps = 0.001
        tvar = SOR(tvar,self.source,self.BC,self.INIT["dx"],epsilon=eps)
    
    def vmax(self):
        return self.INIT["Uc"]


    def display(self,tvar,avar=None):
        L = self.INIT["L"]
        print(np.min(tvar))
        print(np.max(tvar))
        plt.imshow(tvar,cmap=plt.get_cmap('coolwarm'),vmin=np.min(tvar), vmax=np.max(tvar), origin='lower',extent=[0,L,0,L])
        plt.show()
        plt.clf()
        if avar != None:
            pass