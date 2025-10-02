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

x = np.linspace(-3,10,600)
xp = np.zeros_like(x)
tp = np.zeros_like(x)
m = 2
time_scale = 1e-3


# definimos la funcion de propagacion temporal pra una particula libre.

def U(x,t,xp,tp):

     prop = np.sqrt( m/(2*math.pi*1j*(t)))*np.exp(1j*m*(x-xp)**2/(2*(t-tp)))

     return prop

# Definimos la envolvente de la funcion de onda initial en t= 0,

def phi_0(x,xp,p):
    # return np.where((-0.5 <=(x -xp)) & ((x-xp) <= 0.5), np.exp(1j*p*x), 0)
    return np.exp(-(x-xp)**2+1j*p*x)

def phit(x,xp,t,tp):
    phi_t = []
    for i in x:
        phi_t.append(integrate.simpson(phi_0(xp,0, 100)*U(i, t, xp, tp),xp))
    return np.array(phi_t)

# Update function
def update(frame):
    t = 2+ (frame+1) * time_scale
    if  int(t/time_scale) <= 1:
        ax.plot(x,np.abs(phi_0(x,0,10)))
    else:
        phi_t = phit(x,x,t,2)
        ax.cla()  # clear axes
        ax.plot(x, np.abs(phi_t / np.max(np.abs(phi_t))))

    # Re-apply limits and labels (since cla() resets them)
    ax.set_xlim(-3, 10)
    ax.set_ylim(-1, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("|ψ(x,t)|")
    return []

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()


# phi_t = phit(x,x,0.1,0)
# ax.plot(x,np.abs(phi_t/np.max(np.abs(phi_t))))

ani = animation.FuncAnimation(fig, update, frames=1000, interval=20, blit=False)

plt.show()
