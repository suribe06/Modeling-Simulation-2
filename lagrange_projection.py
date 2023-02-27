import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sym
import time

def u(x):
    return np.sin(x)

def plotPhis(phis, a, b):
    plt.clf()
    x_values = np.linspace(a, b, 500)
    for i in range(len(phis)):
        y = [phis[i](xi) for xi in x_values]
        plt.plot(x_values, y, label=f'$\phi_{i}$')
    plt.legend()
    plt.show()

def phi_i(x, i, Omega_j):
    x_min = Omega_j[0]
    x_max = Omega_j[-1]
    if x_min <= x and x <= x_max:
        p = 1
        n = len(Omega_j)
        for k in range(n):
            if k != i: p *= (x - Omega_j[k])/(Omega_j[i] - Omega_j[k])
        return p
    else:
        return 0
    
def dotProduct(f1, f2, a, b):
    return quad(lambda x: f1(x) * f2(x), a, b)[0]
    
def lagrangeBasis(p_degree, n_Omegas, a, b):
    n_points = p_degree * n_Omegas + 1
    points = np.linspace(a,b, n_points)
    Omegas = []
    for i in range(n_Omegas):
        omega = points[i*p_degree : (i+1)*p_degree+1]
        Omegas.append(omega)
    phis_dict = {}
    for j in range(len(Omegas)):
        Omega_j = Omegas[j]
        for k in range(len(Omega_j)):
            x = Omega_j[k]
            phi = lambda x, Omega_j=Omega_j, k=k: phi_i(x, k, Omega_j)
            if x in phis_dict:
                phis_dict[x].append(phi)
            else:
                phis_dict[x] = [phi]
    phis = []
    for x_i, phi_list in phis_dict.items():
        phi_sum = None
        if len(phi_list) == 1:
            phis.append(phi_list[0])
        else:
            phi_sum = phi_list[0]
            for i in range(1, len(phi_list)):
                phi_sum = lambda x, f=phi_sum, g=phi_list[i]: f(x) + g(x)
            phis.append(phi_sum)
    return phis

def lagrangeProjection(p_degree, n_Omegas, a, b):
    n_points = p_degree * n_Omegas + 1
    phis = lagrangeBasis(p_degree, n_Omegas, a, b)
    #plotPhis(phis, a, b)
    B = np.zeros(n_points, dtype=float)
    A = np.zeros((n_points, n_points), dtype=float)
    for i in range(n_points):
        B[i] = dotProduct(u, phis[i], a, b)
        for j in range(i, n_points):
            A[i][j] = A[j][i] = dotProduct(phis[i], phis[j], a, b)
    C = np.linalg.solve(A, B)
    y_proj = lambda x: sum([C[i] * phis[i](x) for i in range(n_points)])
    return y_proj

def testMethod(p_degree, n_Omegas, a, b):
    y_proj = lagrangeProjection(p_degree, n_Omegas, a, b)
    x_values = np.linspace(a, b, 1000)
    y_u = [u(x_i) for x_i in x_values]
    y_proj = [y_proj(x_i) for x_i in x_values]
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_proj, label='$\sum^n_{i=0} c_i \phi_i(x)$')
    plt.plot(x_values, y_u, label='$u(x)$')
    plt.title(f'$u(x)$ projection using lagrange polynomials of degree {p_degree} and {n_Omegas} $\Omega_i$ intervals')
    plt.legend()
    plt.show()

def errorStudy(p_degree, a, b, n_max):
    n_array = [x for x in range(1, n_max)]
    errors = []
    for n in n_array:
        y_proj = lagrangeProjection(p_degree, n, a, b)
        e = np.sqrt(quad(lambda x: (u(x) - y_proj(x))**2, a, b, limit=100)[0])
        errors.append(e)
    p = np.polyfit(n_array, errors, 1)
    plt.scatter(n_array, errors, label='$|| u(x) - \sum^n_{i=0} c_i \phi_i(x)||$')
    plt.plot(n_array, np.polyval(p, n_array), label=f'y = {p[0]:.3f}x + {p[1]:.3f}', color='red')
    plt.title(f'Study of Error Using Lagrange Polynomials of Degree {p_degree}')
    plt.xlabel('n')
    plt.ylabel('error')
    plt.yscale('log')
    plt.legend()
    plt.show()

testMethod(p_degree=2, n_Omegas=6, a=0, b=np.pi)
errorStudy(p_degree=2, a=0, b=np.pi, n_max=13)