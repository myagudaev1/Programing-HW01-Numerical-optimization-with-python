import numpy as np

def example_1(x):
    Q = np.array([[1, 0],[0, 1]])
    f = 0.5 * np.dot(np.dot(x.T,Q), x)
    gradient = np.dot(Q,x)
    hessian = Q
    return f, gradient, hessian

def example_2(x):
    Q = np.array([[1, 0], [0, 100]])
    f = 0.5 * np.dot(np.dot(x.T,Q),x)
    gradient = np.dot(Q, x)
    hessian = Q
    return f, gradient, hessian

def example_3(x):
    temp = (-25 * np.sqrt(3)) + ((1/4) * np.sqrt(3))
    Q = np.array([[75.25, temp], [temp, 25.75]])
    f = 0.5 * np.dot(np.dot(x.T,Q),x)
    gradient = np.dot(Q, x)
    hessian = Q
    return f, gradient, hessian

def rosenbrock(x):
    f = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    gradient = np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 * (x[1] - x[0]**2)])
    hessian = np.array([[2 - 400 * x[1] + 1200 * x[0]**2, -400 * x[0]], [-400 * x[0], 200]])
    return f, gradient, hessian

def linear(x):
    alpha = np.array([1,5])
    f = np.dot(alpha,x)
    gradient = alpha
    hessian = np.zeros((2,2))
    return f, gradient, hessian

def last_example(x):
    f = np.exp(x[0] + 3 * x[1]) + np.exp(x[0] - 3 * x[1]) + np.exp(-x[0])
    gradient = np.array([np.exp(x[0] + 3 * x[1]) + np.exp(x[0] - 3 * x[1]) - np.exp(-x[0]), 3 * np.exp(x[0] + 3 * x[1]) - 3 * np.exp(x[0] - 3 * x[1])])
    hessian = np.array([[np.exp(x[0] + 3 * x[1]) + np.exp(x[0] - 3 * x[1]) + np.exp(-x[0]), 3 * np.exp(x[0] + 3 * x[1]) - 3 * np.exp(x[0] - 3 * x[1])],
                     [3 * np.exp(x[0] + 3 * x[1]) - 3 * np.exp(x[0] - 3 * x[1]), 9 * np.exp(x[0] + 3 * x[1]) + 9 * np.exp(x[0] - 3 * x[1])]])
    return f, gradient, hessian


