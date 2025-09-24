import numpy as np
import matplotlib.pyplot as plt

# Create a sphere parameterization
u = np.linspace(0, np.pi/2, 400)   # angle around z-axis
v = np.linspace(0, np.pi/2, 400)       # angle from z-axis

# Sphere of radius r
r = 1
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones_like(u), np.cos(v))

# Plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, alpha=0.7)

# Equal aspect ratio
ax.set_box_aspect([1,1,1])
plt.show()