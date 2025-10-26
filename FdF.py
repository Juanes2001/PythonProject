import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sympy.assumptions.predicates.order import ExtendedNonZeroPredicate

# Physical constants
wavelength = 633e-9          # 633 nm
k = 2 * np.pi / wavelength   # wave number
kdir = k*np.array([[1,0,0]]).T
omega = 3e8 * k              # angular frequency
E0 = 1.0                     # amplitude
H0 = 0.3

no = 1.6557
ne = 1.4849


time_scale = 1e-16
# Grid: points along z-axis
n_points = 200
x = np.linspace(0, 1000*wavelength, n_points)
y = np.linspace(0, 1000*wavelength, n_points)
z = np.linspace(0, 1000*wavelength, n_points)

r = np.array([z,y,x])


# Coordinates: all vectors are at (0,0,z)
Z = z / wavelength
Y = y / wavelength
X = x / wavelength


xplane = np.linspace(-3, 3, 20)
yplane = np.linspace(-3, 3, 10)
Xplane,Yplane = np.meshgrid(xplane, yplane)
Zplane = np.zeros_like(Yplane) + 200  # Plane at x = 0


# Setup figure
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# --- Plot static YZ plane ---
plane1 = ax.plot_surface(Zplane, Xplane, Yplane, alpha=0.3, color="cyan")
plane2 = ax.plot_surface(Zplane + 200, Xplane, Yplane, alpha=0.3, color="cyan")

ax.set_xlim(0, Z.max() + 200)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xlabel("Z (propagation)")
ax.set_ylabel("X")
ax.set_zlabel("Y (polarization)")

# Initial quiver (dummy, will be updated)
quiver = []
q1 = ax.quiver(np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), color='r')
q2 = ax.quiver(np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), color='b')
q3 = ax.quiver(np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), color='r')
q4 = ax.quiver(np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), color='b')


quiver.extend([q1,q2,q3,q4])



"""
Definimos sobre el espacio las superficies de k en donde se puede definir la dirección del vector de poynting

Definimos el eje optico desde (3,3) y (-3,-3), luego sobre este definimos la esfera y el elipsoide de k para hallar los modos normales
Por lo que aplicamos una rotacion sobre la maya de tal forma que el eje z este en direccion (3,3)-(-3,-3), por lo que hayamos un plano normal a este vector de tal forma que
, contenga a los vectores x y y principales, y ademas que Y este alineado de tal forma que este perpendicular al eje X real.

"""

# Parameters of ellipsoid
a, b, c = k*ne, k*ne, k*no

# Create meshgrid
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)

# Parametric equations
xel = a * np.cos(u) * np.sin(v)
yel = b * np.sin(u) * np.sin(v)
zel = c * np.cos(v)

xef = k*no * np.cos(u) * np.sin(v)
yef = k*no * np.sin(u) * np.sin(v)
zef = k*no * np.cos(v)

# Definimos la matriz de transformación entre los ejes principales del cristal y los ejes normales que manejamos

zrot,xrot,yrot = np.array([[-200,6,6]]).T, np.array([[600,-18,-39982]]).T, np.array([[6,200,0]]).T
zrot,xrot,yrot = zrot / np.linalg.norm(zrot),xrot/ np.linalg.norm(xrot),yrot/ np.linalg.norm(yrot)


mrot = np.column_stack((zrot,xrot,yrot)).T

# Para hallar los indices de refracción solo basta con

P1 = np.array([400, -3,-3 ])   # start
P2 = np.array([200, 3, 3])   # end

# Parameter t
t = np.linspace(0, 1, 100)

# Parametric equations
xline = P1[0] + t * (P2[0] - P1[0])
yline = P1[1] + t * (P2[1] - P1[1])
zline = P1[2] + t * (P2[2] - P1[2])

ax.plot(xline, yline, zline, color="red", linestyle="--", linewidth=2, label="Dashed line")

# Electric field function
def E_field(k,r,t,phi):
    sh = np.shape(z)
    phase = np.zeros(sh[0])
    phase[0:int(t/time_scale)] = np.cos(k.T@r[:,0:int(t/time_scale)] - omega*t + phi)
    return E0 * phase

# Magnetic field function
def H_field(k,r,t,phi):
    sh = np.shape(z)
    phase = np.zeros(sh[0])
    phase[0:int(t/time_scale)] = np.cos(k.T@r[:,0:int(t/time_scale)] - omega*t + phi)
    return H0 * phase


# Update function
def update(frame):
    global quiver
    phi1 = 0
    phi2 = 0
    t = frame * time_scale

    if Z[int(t/time_scale)] < 200:
        Ex = E_field(kdir,r,t,phi1)
        Ey = E_field(kdir,r,t, phi2)
        Ez = np.zeros_like(z)
        Dx = 0.3*Ex
        Dy = 0.3*Ey
        Dz = 0.3 * Ez
        Hx = H_field(kdir, r, t, phi1)
        Hy = H_field(kdir, r, t, phi2)
        Hz = np.zeros_like(z)


        X = np.zeros_like(z)
        Y = np.zeros_like(z)


        ue = np.array([[Ez[int(t/time_scale)-1],Ex[int(t/time_scale)-1],Ey[int(t/time_scale)-1]]])
        uh = np.array([[Hz[int(t/time_scale-1)],-Hy[int(t/time_scale-1)],Hx[int(t/time_scale-1)]]])
        poynting_dir = np.cross(ue[0], uh[0])
        if np.linalg.norm(poynting_dir) != 0:
            poynting_dir = poynting_dir /np.linalg.norm(poynting_dir)


        # Remove old quivers
        for q in quiver:
            q.remove()
        quiver = []

        # Add new ones
        q1 = ax.quiver(Z, X, Y, Ez, Ex, Ey, color='r')
        q2 = ax.quiver(Z, X, Y, Dz, Dx, Dy, color='c')

        quiver.extend([q1, q2])
        return quiver
    elif 200<=Z[int(t/time_scale)] < 400:
        Ex = E_field(kdir, r, t, phi1)
        Ey = E_field(kdir, r, t, phi2)
        Ez = np.zeros_like(z)
        Dx = 0.3 * Ex
        Dy = 0.3 * Ey
        Dz = 0.3 * Ez
        Hx = H_field(kdir, r, t, phi1)
        Hy = H_field(kdir, r, t, phi2)
        Hz = np.zeros_like(z)

        X = np.zeros_like(z)
        Y = np.zeros_like(z)

        ue = np.array([[Ez[int(t / time_scale) - 1], Ex[int(t / time_scale) - 1], Ey[int(t / time_scale) - 1]]])
        uh = np.array([[Hz[int(t / time_scale - 1)], -Hy[int(t / time_scale - 1)], Hx[int(t / time_scale - 1)]]])
        poynting_dir = np.cross(ue[0], uh[0])
        if np.linalg.norm(poynting_dir) != 0:
            poynting_dir = poynting_dir / np.linalg.norm(poynting_dir)

        # Remove old quivers
        for q in quiver:
            q.remove()
        quiver = []

        # Add new ones
        q1 = ax.quiver(Z, X, Y, Ez, Ey, Ex, color='r')
        q2 = ax.quiver(Z, X, Y, Dz, Dy, Dx, color='c')

        quiver.extend([q1, q2])
        return quiver


ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.show()
