import numpy as np

def gradient_descent(f, x0, obj_tol, param_tol, max_iter, grad_f):
    x = np.array(x0, dtype= float)
    path = [x]

    for _ in range(max_iter):
        gradient = grad_f(x)
        if np.linalg.norm(gradient) < obj_tol:
            return x, f(x), True, path
        direction = -gradient
        alpha = 1.0
        beta = 0.8
        while f(x + alpha*direction) > f(x) + 1e-4 * alpha * np.dot(direction,direction):
            alpha *= beta
        
        x += alpha * direction
        path.append(x.copy())
        if np.linalg.norm(alpha * direction) < param_tol:
            return x, f(x), True, path
    
    return x, f(x), False, path

def newton_method(f, x0, obj_tol, param_tol, max_iter, grad_f, hess_f):
    x = np.array(x0, dtype= float)
    path = [x]

    for _ in range(max_iter):
        gradient = grad_f(x)
        hessian = hess_f(x)
        if np.linalg.norm(gradient) < obj_tol:
            return x, f(x), True, path
        if np.linalg.cond(hessian) > 1/1e-10:
            return x, f(x), False, path
        
        direction = -np.linalg.solve(hessian, gradient)

        alpha = 1.0
        beta = 0.8
        while f(x + alpha*direction) > f(x) + 1e-4 * alpha * np.dot(direction,direction):
            alpha *= beta

        x += alpha * direction
        path.append(x.copy())
        if np.linalg.norm(alpha * direction) < param_tol:
            return x, f(x), True, path

    return x, f(x), False, path


def optimization(f, x0, obj_tol, param_tol, max_iter, opt_method = 'gradient_descent', grad_f = None, hess_f = None):
    if opt_method == 'gradient_descent':
        return gradient_descent(f, x0, obj_tol, param_tol, max_iter, grad_f)
    elif opt_method == 'newton_method':
        return newton_method(f, x0, obj_tol, param_tol, max_iter, grad_f, hess_f)
    else:
        raise ValueError(f"Wrong Method {opt_method}")
