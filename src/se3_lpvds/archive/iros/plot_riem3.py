import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the range of x and y
x = np.linspace(-7, 7, 100)
y = np.linspace(-7, 7, 100)

# Create a meshgrid from x and y
X, Y = np.meshgrid(x, y)

# Define the equation of the curved surface (paraboloid)
Z = - 1/25 * (X**2 + Y**2)

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z,  color='lightsteelblue')



# Create a tangent plane
point = np.array([0, 0, 1])  # Center of the sphere
normal = np.array([0, 0, 1])  # Normal vector to the tangent plane


# Create a grid of points for the tangent plane
xx, yy = np.meshgrid(np.linspace(-8, 8, 20), np.linspace(-8, 8, 20))
d = -point.dot(normal)
zz = (-normal[0] * xx - normal[1] * yy - d) * 0.0


ax.plot_surface(xx, yy, zz, color='gray', alpha=0.6, linewidth=0)


ax.scatter(0,0,0,
          color='black')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_title('Curved Surface: Paraboloid')


ax.axis('equal')


ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.axis('off')


plt.savefig('riem.png', dpi=600, transparent=True)


# Show the plot
plt.show()
