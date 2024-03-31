import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a unit sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Create a tangent plane at the top point of the sphere
tangent_point = np.array([0, 0, 1])  # Coordinates of the top point
tangent_normal = np.array([0, 0, 1])  # Normal vector of the tangent plane

# Create a grid for the tangent plane
xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
zz = tangent_normal[0] * xx + tangent_normal[1] * yy + tangent_normal[2]

# Create a vector on the sphere starting from the origin
sphere_vector_start = np.array([0, 0, 0])
sphere_vector_end = np.array([0.5, 0.5, 0.5])  # Change the endpoint as desired

# Create a vector on the tangent plane starting from the point of tangency
tangent_vector_start = tangent_point
tangent_vector_end = np.array([0.3, 0.3, 0])  # Change the endpoint as desired

# Create a 3D plot with a fixed aspect ratio
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the unit sphere
ax.plot_surface(x, y, z, color='b', alpha=0.5)

# Plot the tangent plane
ax.plot_surface(xx, yy, zz, color='r', alpha=0.5)

# Plot the vector on the sphere
ax.quiver(sphere_vector_start[0], sphere_vector_start[1], sphere_vector_start[2],
          sphere_vector_end[0], sphere_vector_end[1], sphere_vector_end[2],
          color='g', label='Sphere Vector')

# Plot the vector on the tangent plane
ax.quiver(tangent_vector_start[0], tangent_vector_start[1], tangent_vector_start[2],
          tangent_vector_end[0], tangent_vector_end[1], tangent_vector_end[2],
          color='m', label='Tangent Vector')

# Hide grid and axes
ax.grid(False)
ax.axis('off')

# Set aspect ratio to 'equal' to ensure a spherical appearance
ax.set_box_aspect([1, 1, 1])

# Set axis limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Show the legend
# ax.legend()

# Show the plot
plt.show()
