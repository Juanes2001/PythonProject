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
import matplotlib.animation as animation


lam = np.linspace(300e-9,700e-9,10000)
nh = 1
p = 312.5e-9
lameav = np.linspace(400e-9,700e-9,50)
Eav = (lameav/p)**2
Del_av = 0.06*np.sqrt(Eav)/2


eav =   (500e-9/p)**2
del_av = 0.009*math.sqrt(eav)/2
alfa = lam/p
h = 60e-6

# Onda incidente y onda reflejada
# E_vec_i = 1/math.sqrt(2)*fun.Evec(Ax,Ay,0,omega = 2*math.pi*fun.c/fun.lambda0,dir = 1)
# E_vec_r = 1/math.sqrt(2)*fun.Evec(Ax,-Ay,0,omega = 2*math.pi*fun.c/fun.lambda0, dir = -1)

#

fig, ax1 = plt.subplots()
data = (fun.find_amplitudes_LCP_RCP(lam,nh,h,p,eav,del_av,'LCP')) *100
ax1.plot(lam*1e9,data)
ax1.plot(lam*1e9,np.ones_like(lam)*100)

ax1.text(
    0.02, 0.98,
    f"h = {h*1e6 }um\nε_av = {round(eav,3)}\nn = {nh}",
    transform=ax1.transAxes,
    verticalalignment="top",
    fontsize=15,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black",
        alpha=0.85
    )
)



# print(data)

# Axis labels
plt.xlabel("Wavelength (nm)",fontsize=15)
plt.ylabel("% of Reflection",fontsize=15)

# Title
plt.title(f"δ = {round(del_av,3)}",fontsize= 18)


# line_module, = ax1.plot([], [], lw=2, color = 'blue')
# frame_text = ax1.text(0.02, 0.9, '', transform=ax1.transAxes, fontsize=12, color='red')
#
# def init():
#     line_module.set_data([],[])
#     frame_text.set_text('')
#     return line_module, frame_text
#
# def animate(frames):
#     y_vals = fun.find_amplitudes_LCP_RCP(lam,nh,h,p,Eav[frames],Del_av[frames],'LCP') *100  # arguments: (x,t,p0,x0,t0)
#     line_module.set_data(lam*1e9, y_vals)
#     frame_text.set_text(f"{frames}")
#     return line_module,frame_text
#
#
# ani = animation.FuncAnimation(fig, animate,init_func=init, frames=50, interval=100, blit=True)

# Optional grid for readability

plt.grid()
plt.show()