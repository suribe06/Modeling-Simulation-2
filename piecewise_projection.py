from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import statistics as st

def u(x):
    return np.sin(x)

def dotProduct(f1, f2, a, b):
    return quad(lambda x: f1(x) * f2(x), a, b)[0]

def manualIntegration(i, j, x_nodes):
    n = len(x_nodes) - 1
    if i == j:
        if i == 0: return (2/6) * abs(x_nodes[i] - x_nodes[i+1])
        elif i == n: return (2/6) * abs(x_nodes[i] - x_nodes[i-1])
        elif i != 0 and i != n: return (2/3) * abs(x_nodes[i] - x_nodes[i+1])
    else:
        return (1/6) * abs(x_nodes[i] - x_nodes[j])

def phi_i(x, x_nodes, i):
    n = len(x_nodes) - 1
    if i == 0:
        return (x-x_nodes[1])/(x_nodes[0]-x_nodes[1]) if x >= x_nodes[0] and x < x_nodes[1] else 0
    elif i == n:
        return (x-x_nodes[n-1])/(x_nodes[n]-x_nodes[n-1]) if x >= x_nodes[n-1] and x <= x_nodes[n] else 0
    else:
        return (x-x_nodes[i-1])/(x_nodes[i]-x_nodes[i-1]) if x >= x_nodes[i-1] and x <= x_nodes[i] else (x-x_nodes[i+1])/(x_nodes[i]-x_nodes[i+1]) if x >= x_nodes[i] and x <= x_nodes[i+1] else 0

def getPiecewiseProjection(x_nodes, n, a, b):
    phi = [lambda x, i=i: phi_i(x, x_nodes, i) for i in range(n+1)]
    B = [dotProduct(u, phi[i], a, b) for i in range(n+1)]
    A = np.zeros((n+1, n+1), dtype=float)
    for i in range(n+1):
        for j in range(n+1):
            if i == j or i == j + 1 or i == j - 1:
                #A[i][j] = dotProduct(phi[i], phi[j], a, b)
                A[i][j] = manualIntegration(i, j, x_nodes)
    C = np.linalg.solve(A, B)
    return C

a, b, N = 0, np.pi, 1000
n = 3
x_nodes = np.linspace(a, b, n+1)

"""
#Plot phi functions
phi = [lambda x, i=i: phi_i(x, x_nodes, i) for i in range(n+1)]
x = np.linspace(a, b, N)
for i in range(n+1):
    y = [phi[i](xi) for xi in x]
    plt.plot(x, y, label=f'phi_{i}')
y_u = [u(x_i) for x_i in x]
plt.plot(x, y_u, label='sin')
plt.legend()
plt.show()
"""

#Plot projection
ans = getPiecewiseProjection(x_nodes, n, a, b)
x = np.linspace(a, b, N)
y_u = [u(x_i) for x_i in x]
y_func = lambda x : sum([ ans[i] * phi_i(x, x_nodes, i) for i in range(n+1)])
y_proj = [y_func(x_i) for x_i in x]
plt.clf()
plt.plot(x, y_u, label='sin')
plt.plot(x, y_proj, label='projection')
plt.legend()
plt.show()

error = [abs(y_u[i] - y_proj[i]) for i in range(N)]
mean_error = st.mean(error)
print(mean_error)
