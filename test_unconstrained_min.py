import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from examples import example_1, example_2, example_3, rosenbrock, linear, last_example
from unconstrained_min import optimization
from utils import plot_contours, plot_values

class TesstingUncostrainedOptimization(unittest.TestCase):
    def setUp(self) -> None:
        if self._testMethodName == "test_rosenbrock_function":
            self.x0 = [-1, 2]
        else:
            self.x0 = [1, 1]
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 1000

    def run_optimization(self,f,gradient,hessian):
        gradient_descent = optimization(f,self.x0,self.obj_tol,self.param_tol,self.max_iter,'gradient_descent', gradient)
        newton_method = optimization(f,self.x0,self.obj_tol,self.param_tol,self.max_iter,'newton_method', gradient, hessian)

        self.assertTrue(gradient_descent[2], 'Gradient Descent Failed')
        if newton_method:
            self.assertTrue(newton_method[2], 'Newton Method Failed')
        
        limits = [-2, 2, -2, 2]
        paths = [(gradient_descent[3], 'Gradient Descent')]
        if newton_method:
            paths.append((newton_method[3], "Newton Method"))
        
        
        plot_contours(f, limits[:2], limits[2:], paths=paths)
        plot_values (
            (gradient_descent[3], "Gradient Descent"),
            (newton_method[3], "Newton's Method") if newton_method else None
        )

    def example_1_test(self):
        f = lambda x:example_1(x)[0]
        gradient = lambda x: example_1(x)[1]
        hessian = lambda x: example_1(x)[2]
        self.run_optimization(f, gradient,hessian)
    
    def test_example_2(self):
        f = lambda x: example_2(x)[0]
        grad = lambda x: example_2(x)[1]
        hess = lambda x: example_2(x)[2]
        self.run_optimization(f, grad, hess)

    def test_example_3(self):
        f = lambda x: example_3(x)[0]
        grad = lambda x: example_3(x)[1]
        hess = lambda x: example_3(x)[2]
        self.run_optimization(f, grad, hess)

    def test_rosenbrock_function(self):
        f = lambda x: rosenbrock(x)[0]
        grad = lambda x: rosenbrock(x)[1]
        hess = lambda x: rosenbrock(x)[2]
        self.run_optimization(f, grad, hess)

    def test_linear_function(self):
        f = lambda x: linear(x)[0]
        grad = lambda x: linear(x)[1]
        hess = lambda x: linear(x)[2]
        self.run_optimization(f, grad, hess)

    def test_last_function(self):
        f = lambda x: last_example(x)[0]
        grad = lambda x: last_example(x)[1]
        hess = lambda x: last_example(x)[2]
        self.run_optimization(f, grad, hess)

if __name__ == '__main__':
    unittest.main()