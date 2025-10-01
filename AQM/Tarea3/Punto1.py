"""
    Este código de programación pretende simular la propagacion temporal de una particula libre en una cierta ventana de
    tiempo
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import integrate

# Definimos las condiciones iniciales de tiempo y espacio

x = np.linspace(-2,10,100)
xp = np.zeros_like(x)
tp = np.zeros_like(x)
m = 2
time_scale = 1e-16


# definimos la funcion de propagacion temporal pra una particula libre.

def U(x,t,xp,tp):

     prop = math.sqrt( m/(2*math.pi*1j*(t-tp)))*np.exp(1j*m*(x-xp)**2/(2*(t-tp)))

     return prop

# Definimos la envolvente de la funcion de onda initial en t= 0,

def phi_0(x,xp):
    return np.where((-0.5 <= (x - xp)) & ((x - xp) <= 0.5), 1, 0)

# Update function
def update(frame):
    t = frame * time_scale
    phi_t = integrate.simpson(phi_0(x,0),U(x,t,0,0))
    ev = ax.plot(x,phi_t)
    return ev

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.show()
