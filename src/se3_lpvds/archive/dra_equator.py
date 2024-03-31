import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a sphere
r = 1  # radius
phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]  # angles for spherical coordinates
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

# Create a tangent plane
point_on_plane = np.array([0, 0, r])  # arbitrary point on the sphere, with z-coordinate r to be on top
normal_to_plane = np.array([0, 0, 1])  # normal vector to the plane, parallel to the z-axis

# Plot the sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere surface
ax.plot_surface(x, y, z, color='gray', alpha=0.5)

# Plot the dashed line representing the equator
theta_equator = np.linspace(0, 2 * np.pi, 100)
x_equator = r * np.cos(theta_equator)
y_equator = r * np.sin(theta_equator)
z_equator = np.zeros_like(theta_equator)
ax.plot(x_equator, y_equator, z_equator, color='k', linestyle='--')

# Set aspect ratio
ax.set_box_aspect([1, 1, 1])  # aspect ratio
ax.set_axis_off()

ax.set_frame_on(False)

# Set aspect ratio
ax.set_box_aspect([1, 1, 1])  # aspect ratio
ax.set_aspect("equal", adjustable="box")

plt.show()
