"""
    Este código de programación pretende simular la propagacion temporal de una particula libre en una cierta ventana de
    tiempo
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as int

# definimos los parametros a usar
x0 = 0
t0 = 0
p0 = 4

hbar = 1
me = 1
sigma = 0.6

# Ahora definimos el propagador y la funcion de onda inicial.

def U(x,t,xp,tp):
    u = np.sqrt(me/(2*math.pi*hbar*1j*(t-tp)))*np.exp(1j*me*(x-xp)**2/(2*hbar*(t-tp)))
    return u

def phi_0(xp):
    Wave_in = 1/(math.sqrt(2*math.pi)*sigma) * np.exp(-(xp-x0)**2/(2*sigma) + 1j*p0*xp/hbar  )
    # Wave_in = np.where(np.abs(xp-x0) < 0.5,np.exp(1j*p0*xp/hbar ),0)
    return Wave_in

# ahora definimos la propagacion al integrar, por lo que definimos los dominios

xvals = np.linspace(-6,50,500)

def phi_t (x,t):
    prop = []
    for i in xvals:
        prop.append(int.simpson(phi_0(xvals)*U(i,t,xvals,t0),xvals))
    return np.array(prop)



fig, ax = plt.subplots()
line_module, = ax.plot([], [], lw=2, color = 'blue')
line_real, = ax.plot([], [], lw=2)
line_imag, = ax.plot([], [], lw=2)
line_module_neg, = ax.plot([], [], lw=2, color='blue')

ax.set_xlim(-5, 50)
ax.set_ylim(-1, 1)

def init():
    line_module.set_data([],[])
    line_real.set_data([], [])
    line_imag.set_data([], [])
    line_module_neg.set_data([], [])
    return line_module,line_real,line_imag,line_module_neg

def animate(i):
    t_val = (i+1) * 0.15
    y_vals = phi_t(xvals, t_val)  # arguments: (x,t,p0,x0,t0)
    line_module.set_data(xvals, np.abs(y_vals))  # real part
    line_real.set_data(xvals, np.real(y_vals))  # real part
    line_imag.set_data(xvals, np.imag(y_vals))  # real part
    line_module_neg.set_data(xvals, -np.abs(y_vals))  # real part
    return line_module,line_real,line_imag,line_module_neg

ani = animation.FuncAnimation(fig, animate,init_func=init, frames=200, interval=50, blit=True)

plt.grid()
plt.show()