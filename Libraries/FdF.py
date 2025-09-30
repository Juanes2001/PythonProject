import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sympy.assumptions.predicates.order import ExtendedNonZeroPredicate

# Physical constants
wavelength = 633e-9          # 633 nm
k = 2 * np.pi / wavelength   # wave number
omega = 3e8 * k              # angular frequency
E0 = 1.0                     # amplitude
H0 = 0.3

time_scale = 1e-16
# Grid: points along z-axis
n_points = 200
z = np.linspace(0, 1000*wavelength, n_points)

# Coordinates: all vectors are at (0,0,z)
Z = z / wavelength
Y = z / wavelength
X = z / wavelength


xplane = np.linspace(-3, 3, 20)
yplane = np.linspace(-3, 3, 10)
Xplane,Yplane = np.meshgrid(xplane, yplane)
Zplane = np.zeros_like(Yplane) + 100  # Plane at x = 0


# Setup figure
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# --- Plot static YZ plane ---
plane1 = ax.plot_surface(Zplane, Yplane, Xplane, alpha=0.3, color="cyan")
plane2 = ax.plot_surface(Zplane + 200, Yplane, Xplane, alpha=0.3, color="cyan")

ax.set_xlim(0, Z.max() + 200)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xlabel("Z (propagation)")
ax.set_ylabel("Y")
ax.set_zlabel("X (polarization)")

# Initial quiver (dummy, will be updated)
quiver = []
q1 = ax.quiver(Z, np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), color='r')
q2 = ax.quiver(Z, np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), color='b')


quiver.extend([q1, q2])



"""
Definimos sobre el espacio las superficies de k en donde se puede definir la dirección del vector de pointing
    

"""

# Electric field function
def E_field(k,t,phi):
    sh = np.shape(z)
    phase = np.zeros(sh[0])



    phase[0:int(t/time_scale)] = np.cos(k.T@z[0:int(t/time_scale)] - omega*t + phi)



    return E0 * phase

# Magnetic field function
def H_field(t,phi):
    sh = np.shape(z)
    phase = np.zeros(sh[0])
    phase[0:int(t/time_scale)] = np.cos(k*z[0:int(t/time_scale)] - omega*t + phi)
    return H0 * phase


# Update function
def update(frame):
    global quiver
    phi = 0
    t = frame * time_scale

    if Z[int(t/time_scale)] < 100 :
        Ex = E_field(t,0)
        Ey = E_field(t, phi)
        Ez = 0
        Dx = 0.3*Ez
        Dy = 0.3*Ey

    ue = np.array([[0,Ey[int(t/time_scale)],Ex[int(t/time_scale)]]]).T

    Hx = H_field(t,0)
    Hy = H_field(t,0)


    # Remove old quivers
    for q in quiver:
        q.remove()
    quiver = []

    # Add new ones
    q1 = ax.quiver(Z, Y, X, np.zeros_like(z), Ey, Ex, color='r')
    q2 = ax.quiver(Z, Y, X, np.zeros_like(z), Dy, Dx, color='b')

    quiver.extend([q1, q2])
    return quiver

ani = animation.FuncAnimation(fig, update, frames=800, interval=50, blit=False)

plt.show()
