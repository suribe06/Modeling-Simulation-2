import numpy as np
import matplotlib.pyplot as plt
import time

def lagrange_interpolation2(x_data, y_data, x):
    n = len(x_data)
    y = 0
    for i in range(n):
        # Calculate the Lagrange basis polynomial for the i-th data point
        l = 1
        for j in range(n):
            if j != i: 
                # Update the Lagrange multiplier
                l *= (x - x_data[j]) / (x_data[i] - x_data[j])
        # Accumulate the value of the Lagrange polynomial at x
        y += y_data[i] * l
    return y

def lagrange_interpolation(x, x_nodes, y_nodes):
    n = len(x_nodes)
    y = np.zeros(len(x))
    for i in range(n):
         # Initialize the Lagrange multiplier with 1
        l = 1.0
        for j in range(n):
            if i != j:
                # Update the Lagrange multiplier
                l *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        # Update the output array with the contribution from the current node
        y += l * y_nodes[i]
    return y

def plot_interpolation(x,y, x_data, y_data):
    plt.plot(x, y, label="Lagrange Interpolation")
    plt.scatter(x_data, y_data, label='Data points', color='red')
    plt.legend()
    plt.show()

x_data = np.linspace(0, 10, 50)
y_data = np.sin(x_data)
x = np.linspace(min(x_data), max(x_data), 500)

# Basic Method
start = time.time()
y = [lagrange_interpolation2(x_data, y_data, xi) for xi in x]
end = time.time()
print("Tiempo de ejecución de la solucion basica:", end - start)
plot_interpolation(x,y, x_data, y_data)

# Efficient method
start = time.time()
y = lagrange_interpolation(x, x_data, y_data)
end = time.time()
print("Tiempo de ejecución de la solucion eficiente:", end - start)
plot_interpolation(x,y, x_data, y_data)