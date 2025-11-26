import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL.ImageColor import colormap
from matplotlib import colors, colors
from matplotlib.pyplot import colormaps
from sympy.codegen.ast import integer
from sympy.vector import CoordSys3D

r = CoordSys3D('r')

nx , ny , alfa = sp.symbols('nx ny alfa', real=True)

eav = (nx**2 + ny**2)/2
del_av = (nx**2 - ny**2)/2

n1_plus = sp.sqrt(alfa**2  + eav + sp.sqrt(del_av**2 + 4*alfa**2*eav))
n1_minus = -sp.sqrt(alfa**2  + eav + sp.sqrt(del_av**2 + 4*alfa**2*eav))
n2_plus = sp.sqrt(alfa**2  + eav - sp.sqrt(del_av**2 + 4*alfa**2*eav))
n2_minus = -sp.sqrt(alfa**2  + eav - sp.sqrt(del_av**2 + 4*alfa**2*eav))


A1 =  1*r.i -sp.I * (nx**2 - n1_plus**2 - alfa**2)/(2*alfa*n1_plus) *r.j
A2 = 1*r.i -sp.I * (nx**2 - n1_minus**2 - alfa**2)/(2*alfa*n1_minus)*r.j
A3 =1*r.i -sp.I * (nx**2 - n2_plus**2 - alfa**2)/(2*alfa*n2_plus)*r.j
A4 = 1*r.i -sp.I * (nx**2 - n2_minus**2 - alfa**2)/(2*alfa*n2_minus)*r.j


print(sp.simplify(sp.conjugate(A1).dot(A3)))



