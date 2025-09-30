import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

wavelength = 633e-9          # 633 nm
k = 2 * np.pi / wavelength   # wave number
kdir = k*np.array([[1,0,0]]).T
omega = 3e8 * k              # angular frequency
E0 = 1.0                     # amplitude
H0 = 0.3

no = 1.6557
ne = 1.4849


# Parameters of ellipsoid
a, b, c = ne, ne, no

# Create meshgrid
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)

# Parametric equations
xel = a * np.cos(u) * np.sin(v)
yel = b * np.sin(u) * np.sin(v)
zel = c * np.cos(v)

xef = no * np.cos(u) * np.sin(v)
yef = no * np.sin(u) * np.sin(v)
zef = no * np.cos(v)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(zel, xel, yel, color="red", alpha=0.6)
ax.plot_surface(zef, xef, yef, color="blue", alpha=0.6)

ax.set_xlabel("Z")
ax.set_ylabel("X")
ax.set_zlabel("Y")

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)

plt.show()