import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functions as fn


"""
Definicion de los operadores L2 , Lx,Ly,Lz en coordenadas esfericas Para ello usaremos Sympy para definir estos operadores

"""
#Definimos parametros a usar en unidades atómicas
hbar = 1
me = 1

r, phi, thet  = sp.symbols('r phi theta', real=True)

# Comencemos con uno sencillo
def Lz_op (f : sp.Function):
    Lz = -1j*hbar*sp.diff(f,phi)
    return Lz
