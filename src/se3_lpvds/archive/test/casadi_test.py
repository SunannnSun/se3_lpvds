import casadi as ca
import numpy as np

# Define the symbolic optimization variable (matrix)
A = ca.MX.sym('A', 3, 3)  # Adjust the matrix size as needed

# Construct the Frobenius norm objective
objective = ca.norm_2(A)

# Create the optimization problem
problem = {'x': A, 'f': objective}

# Solve the optimization problem
solver_opts = {'ipopt': {'print_level': 0}}
solver = ca.nlpsol('solver', 'ipopt', problem, solver_opts)
result = solver(x0=np.random.rand(3))

# Extract the optimal matrix
optimal_matrix = result['x'].full()

print("Optimal Matrix:")
print(optimal_matrix)
