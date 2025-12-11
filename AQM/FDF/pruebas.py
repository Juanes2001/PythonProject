import funciones as fun

import sympy as sp
import sympy.vector as vec
from sympy.vector import CoordSys3D
import numpy as np

from IPython.display import display, Math
import scipy.integrate as integ
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


lam = np.linspace(500e-9,1100e-9,5000)
nh = 1
eav = 1.462**2
del_av = 0.06*math.sqrt(eav)/2
p = 525e-9
alfa = lam/p
h = 60e-6


Ax = sp.Integer(1)
Ay = -sp.I

# Onda incidente y onda reflejada
# E_vec_i = 1/math.sqrt(2)*fun.Evec(Ax,Ay,0,omega = 2*math.pi*fun.c/fun.lambda0,dir = 1)
# E_vec_r = 1/math.sqrt(2)*fun.Evec(Ax,-Ay,0,omega = 2*math.pi*fun.c/fun.lambda0, dir = -1)

#

fig, ax1 = plt.subplots()

data = fun.find_amplitudes_LinPol(lam,nh,sp.pi/4,h,p,eav,del_av)


print(data)

ax1.plot(lam,data)
# ax1.set_ylim(0,1.3)
# ax1.set_xlim(700e-9,800e-9)
plt.grid()
plt.show()



# print(fun.find_amplitudes_LinPol(n1,n2,nh,math.pi,h,"oelo"))