import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='lightsteelblue', edgecolor='none', alpha=0.7, shade=True, rstride=1, cstride=1)

# Create a dashed curve around the equator
theta = np.linspace(0, 2 * np.pi, 100)
# x_equator = 0.98* np.cos(theta)
# y_equator = 0.98*np.sin(theta)
x_equator = np.cos(theta)
y_equator = np.sin(theta)

z_equator = np.zeros_like(theta)

# Plot the dashed curve
ax.plot(x_equator, y_equator, z_equator, linestyle='--', color='black', alpha=1)


# Set equal aspect ratio for all axes


# theta = 4
# ax.scatter(np.cos(theta), np.sin(theta), 0)
# point = np.array([np.cos(theta),  np.sin(theta), 0]) 

# normal = np.array([np.cos(theta),  np.sin(theta), 0])  # Normal vector to the tangent plane
# xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 20), np.linspace(-1.5, 1.5, 20))
# d = -point.dot(normal)
# zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

# ax.plot_surface(xx, yy, zz, color='gray', alpha=0.1, linewidth=0)



ax.axis('equal')


ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.axis('off')


plt.savefig('sphere.png', dpi=600)



plt.show()
