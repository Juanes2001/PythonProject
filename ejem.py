import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL.ImageColor import colormap
from matplotlib import colors, colors
from matplotlib.pyplot import colormaps
from sympy.codegen.ast import integer
from sympy.vector import CoordSys3D
import math

R = CoordSys3D('r')

nx , ny = sp.symbols('nx ny', real=True)

lam = np.linspace(200e-9,900e-9,5000)

k0 = 2*math.pi/lam
nh = 1
eav = 1.462**2
del_av = 0.01*eav/2
p = 500e-9
alfa = lam/p
n2 = np.sqrt(alfa**2 + eav + np.sqrt(del_av**2 + 4*eav*alfa**2+0j)+0j)
h = 1e-2
w = 1j*((eav+del_av)-n2**2-alfa**2)/(2*alfa*n2)

# t=  1/ (1j/2  * np.exp(-1j*k0*nh*h) *( (w-1j)**2*np.exp(-1j*k0*n2*h)- (w+1j)**2*np.exp(1j*k0*n2*h)))
r= (w**2+1)*(1-np.exp(-1j*2*k0*n2*h))/(2*w*(1+np.exp(-1j*2*k0*n2*h))
                                       -1j*(w**2-1)*(1-np.exp(-1j*2*k0*n2*h)) )
# y = 2*w*(1+np.exp(-1j*2*k0*n2*h)) -1j*(w**2-1)*(1-np.exp(-1j*2*k0*n2*h))

# print(np.sqrt((700e-9/p)**2 + eav - np.sqrt(del_av**2 + 4*eav*(700e-9/p)**2+0j)+0j))
fig1, ax1 = plt.subplots()

ax1.plot(lam,1-np.abs(r)**2)
ax1.plot(lam,np.abs(r)**2)
# ax1.plot(lam,np.abs(y))
# ax1.set_xlim(700e-9,800e-9)
# print(np.abs(y))

plt.grid()
plt.show()




