import pandas as pd
from gurobipy import Model, GRB, quicksum
import numpy as np

# Read data from CSV files
customer_locations = pd.read_csv('customer_locations.csv', header=None)
driver_locations = pd.read_csv('driver_locations.csv', header=None)
warehouse_locations = pd.read_csv('warehouse_locations.csv', header=None)
supply = pd.read_csv('supply.csv', header=None)

print(driver_locations.head())
print(warehouse_locations.head())
print(customer_locations.head())


# Sets
D = list(range(len(driver_locations)))  # Drivers
H = ['H' + str(i) for i in D]  # Home locations
M = ['W' + str(i) for i in range(len(warehouse_locations))]  # Warehouses
N = ['C' + str(i) for i in range(len(customer_locations))]  # Customers
V = H + M + N  # All nodes
Q = 5  # Driver capacity

# Positions
positions = {}

# Add driver home positions
for idx, row in driver_locations.iterrows():
    x, y = row[0], row[1]  # Explicitly access the columns by position
    positions[H[idx]] = (x, y)

# Add warehouse positions
for idx, row in warehouse_locations.iterrows():
    x, y = row[0], row[1]  # Explicitly access the columns by position
    positions[M[idx]] = (x, y)

# Add customer positions
for idx, row in customer_locations.iterrows():
    x, y = row[0], row[1]  # Explicitly access the columns by position
    positions[N[idx]] = (x, y)

# Distances
def distance(i, j):
    xi, yi = positions[i]
    xj, yj = positions[j]
    return np.hypot(xi - xj, yi - yj)

# Arcs between nodes (complete graph)
A = [(i, j) for i in V for j in V if i != j]
c = {(i, j): distance(i, j) for (i, j) in A}

# Warehouse supplies
s = {}
for idx, val in supply.iterrows():
    s[M[idx]] = val[0]

# Initialize the model
model = Model('DeliveryOptimization')

# Decision variables
x = model.addVars(D, A, vtype=GRB.BINARY, name='x')
y = model.addVars(D, N, vtype=GRB.BINARY, name='y')
u = model.addVars(D, N, vtype=GRB.INTEGER, lb=1, ub=Q, name='u')

# Objective function
model.setObjective(
    quicksum(c[i, j] * x[d, i, j] for d in D for (i, j) in A),
    GRB.MINIMIZE
)

# Constraints

# Driver departure from home
for d in D:
    h_d = H[d]
    model.addConstr(
        quicksum(x[d, h_d, j] for j in V if (h_d, j) in A) == 1
    )

# Flow conservation
for d in D:
    for i in V:
        if i != H[d]:
            model.addConstr(
                quicksum(x[d, i, j] for j in V if (i, j) in A) -
                quicksum(x[d, j, i] for j in V if (j, i) in A) == 0
            )

# Driver capacity constraints
for d in D:
    model.addConstr(quicksum(y[d, i] for i in N) <= Q)

# Customer assignment
for i in N:
    model.addConstr(quicksum(y[d, i] for d in D) == 1)

# Visit-flow linking constraints
for d in D:
    for i in N:
        model.addConstr(
            y[d, i] == quicksum(x[d, i, j] for j in V if (i, j) in A)
        )

# Warehouse supply constraints
for w in M:
    model.addConstr(
        quicksum(
            x[d, w, j] for d in D for j in V if (w, j) in A
        ) <= s[w]
    )

# Subtour elimination constraints (MTZ)
for d in D:
    for i in N:
        for j in N:
            if i != j:
                if (i, j) in A:
                    model.addConstr(
                        u[d, i] - u[d, j] + Q * x[d, i, j] <= Q - 1
                    )

# Optimize the model
model.optimize()

# Print the delivery schedule
if model.status == GRB.OPTIMAL:
    print('Optimal total distance:', model.objVal)
    for d in D:
        route = []
        # Starting from driver's home
        h_d = H[d]
        current_node = h_d
        route.append(current_node)
        while True:
            next_nodes = [j for j in V if (current_node, j) in A and x[d, current_node, j].X > 0.5]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            route.append(next_node)
            current_node = next_node
        print(f'Driver {d + 1} route: {route}')
else:
    print('No optimal solution found.')