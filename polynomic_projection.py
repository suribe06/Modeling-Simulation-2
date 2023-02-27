from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import statistics as st

def u(x):
    return np.sin(x)

def dotProduct(f1, f2, a, b):
    return quad(lambda x: f1(x) * f2(x), a, b)[0]

def getPolynomicProjection(n, a, b):
    base = [lambda x, i=i: x**i for i in range(n+1)]
    B = [dotProduct(u, base[i], a, b) for i in range(n+1)]
    A = np.zeros((n+1, n+1), dtype=float)
    for i in range(n+1):
        for j in range(n+1):
            A[i][j] = dotProduct(base[i], base[j], a, b)
    C = np.linalg.solve(A, B)
    return C

a, b, N = 0, np.pi, 1000
n = 3
ans = getPolynomicProjection(n, a, b)
x = np.linspace(a, b, N)
y_u = [u(x_i) for x_i in x]
y_func = lambda x : sum([ ans[i] * x**i for i in range(n+1)])
y_proj = [y_func(x_i) for x_i in x]
plt.plot(x, y_u, label='sin')
plt.plot(x, y_proj, label='projection')
plt.legend()
plt.show()

error = [abs(y_u[i] - y_proj[i]) for i in range(N)]
mean_error = st.mean(error)
print(mean_error)