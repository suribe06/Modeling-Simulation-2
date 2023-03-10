import sympy as sym
from scipy.integrate import quad
import numpy as np

x = sym.symbols('x')
base = [ sym.sin(x), sym.sin(2*x), sym.sin(3*x), sym.sin(4*x), sym.cos(x), sym.cos(2*x), sym.cos(3*x), sym.cos(4*x) ]

def dotProduct(f1, f2, a, b):
    x = sym.symbols('x')
    integrand = f1* f2
    return sym.integrate(integrand, (x, a, b)).evalf()

a, b = -np.pi, np.pi
n = len(base)
A = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        A[i][j] = A[j][i] = dotProduct(base[i], base[j], a, b)

f = x**2 + 1
B = [dotProduct(f, base[i], a, b) for i in range(n)]
print(sym.integrate((sym.sin(4*x)*sym.sin(4*x)), (x, a, b)).evalf())