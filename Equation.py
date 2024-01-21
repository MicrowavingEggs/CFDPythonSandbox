from matplotlib import pyplot as plt
import numpy as np
from numba import jit

class Equation():

    def __init__(self):
        self.tvar = None
        self.avar = None

    def display(self):
        raise Exception("Vous ne pouvez pas utiliser la class template pour la méthode display")
    
    def update(self,tvar):
        raise Exception("Vous ne pouvez pas utiliser la class template pour la méthode update")
    
    def vmax(self):
        raise Exception("Vous ne pouvez pas utiliser la class template pour la méthode update")