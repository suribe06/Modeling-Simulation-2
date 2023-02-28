import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import scipy
import time

def plotPhis(phis, a, b):
    x = sym.symbols('x')
    x_values = np.linspace(a, b, 500)
    n = len(phis)
    for i in range(n):
        plt.plot(x_values, sym.lambdify(x, phis[i])(x_values), label=f'$\phi_{i}$')
    plt.legend()
    plt.show()

def dotProduct(f1, f2, a, b):
    x = sym.symbols('x')
    integrand = f1* f2
    return sym.integrate(integrand, (x, a, b)).evalf()

def phi_i(x, i, Omega_j):
    x_min = Omega_j[0]
    x_max = Omega_j[-1]
    n = len(Omega_j)
    p = 1
    for k in range(n):
        if k != i:
            p *= (x - Omega_j[k]) / (Omega_j[i] - Omega_j[k])
    return sym.Piecewise((sym.simplify(p), sym.And(x >= x_min, x <= x_max)), (0, True))

def lagrangeBasis(p_degree, n_Omegas, a, b):
    n_points = p_degree * n_Omegas + 1
    points = np.linspace(a,b, n_points)
    Omegas = []
    for i in range(n_Omegas):
        omega = points[i*p_degree : (i+1)*p_degree+1]
        Omegas.append(omega)
    phis_dict = {}
    x = sym.symbols('x')
    for j in range(len(Omegas)):
        Omega_j = Omegas[j]
        for k in range(len(Omega_j)):
            x_val = Omega_j[k]
            phi = phi_i(x, k, Omega_j)
            if x_val in phis_dict:
                phis_dict[x_val].append(phi)
            else:
                phis_dict[x_val] = [phi]
    phis = []
    for x_val, phi_list in phis_dict.items():
        phi_sum = None
        if len(phi_list) == 1:
            phis.append(phi_list[0])
        else:
            phi_sum = phi_list[0]
            for i in range(1, len(phi_list)):
                phi_sum += phi_list[i]
            phis.append(phi_sum)
    return phis

def calculateSubmatrix(p_degree, phis, a, b):
    p_degree += 1
    A = np.zeros((p_degree, p_degree), dtype=float)
    for i in range(p_degree):
        for j in range(i, p_degree):
            A[i][j] = A[j][i] = dotProduct(phis[i], phis[j], a, b)
    A[p_degree-1][p_degree-1] = A[0][0]
    return A

def lagrangeProjection(u, p_degree, n_Omegas, a, b):
    n_points = p_degree * n_Omegas + 1
    x = sym.symbols('x')
    start = time.time()
    phis = lagrangeBasis(p_degree, n_Omegas, a, b)
    end = time.time()
    print("Tiempo de ejecucion calculo de phis:", end - start)
    #plotPhis(phis, a, b)
    start = time.time()
    B = np.zeros(n_points, dtype=float)
    for i in range(n_points):
        B[i] = dotProduct(u, phis[i], a, b)
    A = np.zeros((n_points, n_points), dtype=float)
    submatrix = calculateSubmatrix(p_degree, phis, a, b)
    l = 0
    for i in range(n_Omegas):
        l = i * p_degree
        A[l:l+p_degree+1, l:l+p_degree+1] += submatrix
    end = time.time()
    print("Tiempo de ejecucion llenado de matriz A y vector B:", end - start)
    print("Condicion de la matriz A", np.linalg.cond(A))
    C = np.linalg.solve(A, B)
    y_proj = sum(C[i]*phis[i] for i in range(n_points))
    return y_proj

def testMethod(u, p_degree, n_Omegas, a, b):
    f_hat = lagrangeProjection(u, p_degree, n_Omegas, a, b)
    x = sym.symbols('x')
    x_values = np.linspace(a, b, 1000)
    y_u = sym.lambdify(x, u)(x_values)
    y_proj = sym.lambdify(x, f_hat)(x_values)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_proj, label='$\sum^n_{i=0} c_i \phi_i(x)$')
    plt.plot(x_values, y_u, label='$u(x)$')
    plt.title(f'$u(x)$ projection using lagrange polynomials of degree {p_degree} and {n_Omegas} $\Omega_i$ intervals')
    plt.legend()
    plt.savefig("example.png")

def errorStudy(u, p_degree, a, b, n_max):
    n_array = [x for x in range(1, n_max+1)]
    errors = []
    for n in n_array:
        f_hat = lagrangeProjection(u, p_degree, n, a, b)
        e = sym.lambdify([x], (f_hat - u)**2)
        L2_norm = np.sqrt(scipy.integrate.quad(e, a, b)[0])
        errors.append(L2_norm)
    errors = np.array(errors)
    n_array = np.array(n_array)
    p = np.polyfit(np.log10(n_array), np.log10(errors), 1)
    x_line = np.logspace(np.log10(n_array[0]), np.log10(n_array[-1]), 100)
    y_line = 10**(p[1]) * x_line**(p[0])
    print(f'La recta que minimiza los puntos es y = {p[0]:.3f}x + {p[1]:.3f}')
    plt.clf()
    plt.scatter(n_array, errors, label='$|| u(x) - \sum^n_{i=0} c_i \phi_i(x)||$')
    plt.plot(x_line, y_line, label=f'y = {p[0]:.3f}x + {p[1]:.3f}', color='red')
    plt.title(f'Study of Error Using Lagrange Polynomials of Degree {p_degree}')
    plt.xlabel('n')
    plt.ylabel('error')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig("errors.png")

x = sym.symbols('x')
u = sym.sin(x)

#testMethod(u, p_degree=4, n_Omegas=10, a=0, b=np.pi)
errorStudy(u, p_degree=2, a=0, b=np.pi, n_max=20)