from methods import *
from DNSPressureSteadyIncompressible import *
from DNSAdvection import *
from utils import *

### DECLARATION
INIT = parmparse()

CFL = INIT["CFL"]

dx = INIT["L"]/(INIT["Nx"]-1)
INIT["dx"] = dx
tstart = INIT["tstart"]
tend = INIT["tend"]

EQUATION = DNSAdvection(INIT)
STEADY = EQUATION.isSteady
### INIT
update = EQUATION.update
tvar = EQUATION.tvar
vmax = EQUATION.vmax()


dt = CFL*dx/vmax
t = 0


### MISC
DISPLAY = INIT["DISPLAY_FLAG"]
COMPUTE_AVAR = INIT["COMPUTE_AVAR_FLAG"]
print(COMPUTE_AVAR)
SAVE = INIT["SAVE_FLAG"]






def main():
    ###CHECK
    #if len(dx) != NDIM:
    #    raise Exception("La dimension de dx n'est pas égale à NDIM : len(dx) = "+str(len(dx))," et NDIM = ",str(NDIM))
    if not(STEADY):
        if tstart == tend:
            raise Exception("Le temps final est inférieur au temps initial.")
        if tend-tstart <= dt:
            raise Exception("dt > tend-tstart. dx est peut-être trop grand.")


    ### MAINLOOP
    if STEADY:
        update(tvar)
    else :
        t = tstart
        while t < tend:
            t += dt
            #print((t-tstart)/(tend-tstart))
            update(tvar,dt,t=t)

    ## Compute avar
    if COMPUTE_AVAR:
        avar = EQUATION.computeAvar(tvar,time=t)
    else:
        avar = None
        
    #Display
    if DISPLAY:
        EQUATION.display(tvar,avar=avar)






if __name__ == '__main__':
    main()