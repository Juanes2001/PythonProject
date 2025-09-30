import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
λ = 633e-9  # wavelength (m)
k = 2 * np.pi / λ
ω = 2 * np.pi * 3e8 / λ

# z positions for arrows
z = np.linspace(0, 5 * λ, 20)
x = np.zeros_like(z)
y = np.zeros_like(z)

# Define multiple waves
waves = [
    {"amplitude": 1.0, "phase": 0, "color": "r"},  # red wave
    {"amplitude": 0.5, "phase": np.pi / 2, "color": "b"}  # blue wave
]

# Setup figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(0, 5 * λ)
ax.set_xlabel("X (polarization)")
ax.set_ylabel("Y")
ax.set_zlabel("Z (propagation)")

# Create a quiver for each wave
quiv_objs = []
for w in waves:
    q = ax.quiver(x, y, z, np.zeros_like(z), np.zeros_like(z), np.zeros_like(z),
                  color=w["color"], length=0.5, normalize=True)
    quiv_objs.append(q)


def update(frame):
    t = frame * 1e-16  # time step

    for q, w in zip(quiv_objs, waves):
        Ex = w["amplitude"] * np.cos(k * z - ω * t + w["phase"])
        # update arrows (E field points in x)
        q.remove()  # remove old arrows
        q = ax.quiver(x, y, z, Ex, np.zeros_like(z), np.zeros_like(z),
                      color=w["color"], length=0.5, normalize=True)
        quiv_objs[waves.index(w)] = q


ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
plt.show()