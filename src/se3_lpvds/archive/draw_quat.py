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

# Plot the tangent plane
xx, yy = np.meshgrid(np.linspace(-1.25, 1.25, 10), np.linspace(-1.25, 1.25, 10))
zz = np.full_like(xx, fill_value=point_on_plane[2])  # constant z-value for the plane
ax.plot_surface(xx, yy, zz, color='k', alpha=0.2)

# Remove background
ax.set_axis_off()

ax.set_frame_on(False)

# Set aspect ratio
ax.set_box_aspect([1, 1, 1])  # aspect ratio
ax.set_aspect("equal", adjustable="box")

plt.show()
