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


lam = np.linspace(300e-9,700e-9,5000)
nh = 1
p = 525e-9
eav =   (p/p)**2
del_av = 0.06*math.sqrt(eav)/2
alfa = lam/p
h = 10e-6


Ax = sp.Integer(1)
Ay = -sp.I

# Onda incidente y onda reflejada
# E_vec_i = 1/math.sqrt(2)*fun.Evec(Ax,Ay,0,omega = 2*math.pi*fun.c/fun.lambda0,dir = 1)
# E_vec_r = 1/math.sqrt(2)*fun.Evec(Ax,-Ay,0,omega = 2*math.pi*fun.c/fun.lambda0, dir = -1)

#

fig, ax1 = plt.subplots()

data = fun.find_amplitudes_LCP_RCP(lam,nh,h,p,eav,del_av,'LCP') *100


print(data)

# Axis labels
plt.xlabel("Wavelength (nm)")
plt.ylabel("% of Reflection")

# Title
plt.title("Reflectance at 525 nm")

# Optional grid for readability

ax1.plot(lam*1e9,data)
# ax1.set_ylim(0,1.3)
# ax1.set_xlim(700e-9,800e-9)
ax1.plot(lam*1e9,np.ones_like(lam)*100)
plt.grid()
plt.show()



# print(fun.find_amplitudes_LinPol(n1,n2,nh,math.pi,h,"oelo"))