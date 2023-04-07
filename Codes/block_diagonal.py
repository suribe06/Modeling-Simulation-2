import numpy as np

def manualIntegration(i, j, x_nodes):
    n = len(x_nodes) - 1
    if i == j:
        if i == 0: return (2/6) * abs(x_nodes[i] - x_nodes[i+1])
        elif i == n: return (2/6) * abs(x_nodes[i] - x_nodes[i-1])
        elif i != 0 and i != n: return (1/3) * abs(x_nodes[i] - x_nodes[i+1])
    else:
        return (1/6) * abs(x_nodes[i] - x_nodes[j])

def calculateSubmatrix(i, x_nodes):
    submatrix = np.zeros((2, 2), dtype=float)
    for j in range(2):
        for k in range(2):
            submatrix[j][k] = manualIntegration(i+j, i+k, x_nodes)
    return submatrix

def fillMatrixByBlocks(n, x_nodes):
    A = np.zeros((n+1, n+1), dtype=float)
    for i in range(n):
        submatrix = calculateSubmatrix(i, x_nodes)
        A[i:i+2, i:i+2] += submatrix
    return A

a, b, N = 0, np.pi, 1000
n = 3
x_nodes = np.linspace(a, b, n+1)
A = fillMatrixByBlocks(n, x_nodes)
print(A)