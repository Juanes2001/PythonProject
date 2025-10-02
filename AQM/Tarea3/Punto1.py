"""
    Este código de programación pretende simular la propagacion temporal de una particula libre en una cierta ventana de
    tiempo
"""
import math

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy.codegen.ast import integer

# definimos los parametros a usar
x0 = 0
t0 = 0
p0 = 30
num = 30

# Ahora definimos un paquete de ondas en forma de rectangulo la cual vamos a propagar libremente

x,xnot, t, tnot = sp.symbols('x xnot t tnot')
n,k = sp.symbols('n k',integer = True, positive = True)

phi_0 = sp.Piecewise(
    (sp.exp(1j*p0*(x-xnot)), ((x-xnot) >= -0.5) & ((x-xnot) <= 0.5)),
          (0, True)
)
phi_0num = sp.lambdify((x,xnot), phi_0, 'numpy')

U = sp.sqrt(1/(2*math.pi*1j*(t-tnot)))*(sp.summation(  (1j*(x-xnot)**2/(2*(t-tnot)))**k/sp.factorial(k), (k, 0, n)))

U_num =  sp.lambdify((x,xnot,t,tnot,n), U, 'scipy')


phi_t = sp.integrate(phi_0num(x,xnot)*U_num(x,xnot,t,t0,n),(xnot,-10,10))
phi_tnum = sp.lambdify((x,t,n), phi_t, 'numpy')

x_vals = np.linspace(-10, 10, 400)


fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-2, 10)
ax.set_ylim(-1, 3)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    t_val = i * 0.1
    y_vals = phi_tnum(x_vals, t_val,num)
    line.set_data(x_vals, y_vals)
    return line,

# 7. Create animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=200, interval=50, blit=True
)
plt.grid()
plt.show()