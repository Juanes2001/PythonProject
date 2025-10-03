"""
    Este código de programación pretende simular la propagacion temporal de una particula libre en una cierta ventana de
    tiempo
"""
import math

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# definimos los parametros a usar
x0 = -10
t0 = 0
p0 = 5
num = 30

# Ahora definimos un paquete de ondas en forma de rectangulo la cual vamos a propagar libremente

x, xnot, t, tnot = sp.symbols('x xnot t tnot', real=True)



phi_0 = sp.exp(-(xnot-x0)**2+sp.I*p0*xnot)


U = sp.sqrt(1/(2*sp.pi*sp.I*(t-t0))) * sp.exp(sp.I*(x-xnot)**2/(2*(t-t0)))


phi_t = sp.integrate(phi_0*U,(xnot,-sp.oo,sp.oo), conds='none')
phi_t = phi_t.simplify()
psi_t_x = sp.simplify(str(phi_t.simplify()).replace("exp_polar", "exp"))
prob_t_x = sp.lambdify((x, t), psi_t_x, modules='numpy')

x_vals = np.linspace(-10, 30, 700)


fig, ax = plt.subplots()
line_module, = ax.plot([], [], lw=2, color = 'blue')
line_real, = ax.plot([], [], lw=2)
line_imag, = ax.plot([], [], lw=2)
line_module_neg, = ax.plot([], [], lw=2, color='blue')

ax.set_xlim(-10, 30)
ax.set_ylim(-1, 1)

def init():
    line_module.set_data([],[])
    line_real.set_data([], [])
    line_imag.set_data([], [])
    line_module_neg.set_data([], [])
    return line_module,line_real,line_imag,line_module_neg

def animate(i):
    t_val = (i+1) * 0.1
    y_vals = prob_t_x(x_vals, t_val)  # arguments: (x,t,p0,x0,t0)
    line_module.set_data(x_vals, np.abs(y_vals))  # real part
    line_real.set_data(x_vals, np.real(y_vals))  # real part
    line_imag.set_data(x_vals, np.imag(y_vals))  # real part
    line_module_neg.set_data(x_vals, -np.abs(y_vals))  # real part
    return line_module,line_real,line_imag,line_module_neg

ani = animation.FuncAnimation(fig, animate,init_func=init, frames=200, interval=50, blit=True)

plt.grid()
plt.show()