import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Create a sphere
r = 1
phi, theta = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = r*np.sin(theta)*np.cos(phi)
y = r*np.sin(theta)*np.sin(phi)
z = r*np.cos(theta)

# Create a tangent plane
point = np.array([0, 0, 1])  # Center of the sphere
normal = np.array([0, 0, 1])  # Normal vector to the tangent plane


# Create a grid of points for the tangent plane
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 20), np.linspace(-1.5, 1.5, 20))
d = -point.dot(normal)
zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]



# Calculate a point on the tangent plane that lies on the sphere's surface
point_on_plane = np.array([0, 0, 1])  # Start from the top (north pole)

# Define a vector direction on the tangent plane
# vector_direction = np.array([-1, -1, 0])  # Example vector direction

ax.scatter(point_on_plane[0], point_on_plane[1], point_on_plane[2],
          color='black')

"""
# Choose a starting and ending point for the vector
start_point = np.array([0, 0])  # Example start point
end_point = np.array([-np.pi/2, -np.pi/2])    # Example end point

# Calculate curved path along the sphere
phi_path = np.linspace(start_point[0], end_point[0], 100)
theta_path = np.linspace(start_point[1], end_point[1], 100)
x_path = r * np.sin(theta_path) * np.cos(phi_path)
y_path = r * np.sin(theta_path) * np.sin(phi_path)
z_path = r * np.cos(theta_path)

# Plot curved vector along the surface
ax.plot(x_path, y_path, z_path, color='r')
"""







# Plot sphere
ax.plot_surface(x, y, z, color='lightsteelblue', edgecolor='none', alpha=0.8, shade=True, rstride=1, cstride=1)

# Plot tangent plane
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.1, linewidth=0)

# Set equal aspect ratio
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax.axis('equal')


ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.axis('off')


plt.savefig('sphere.png', dpi=600)

plt.show()
