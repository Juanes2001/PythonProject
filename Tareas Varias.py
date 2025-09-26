import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Tuple

np.set_printoptions(precision=4)


def _numerical_derivative(f: Callable[[float], float], x: float) -> float:
    h = 1e-6
    return (f(x + h) - f(x - h)) / (2 * h)


def _numerical_second_derivative(
    f: Callable[[float], float], x: float, h: float = 1e-6
) -> float:
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)


def newtonMod(
    f: Callable[[float], float],
    x0: float,
    delta: float = 1e-6,
    epsilon: float = 1e-6,
    m: int = 100,
) -> Tuple[float, float, int, float]:
    """
    Modified Newton's method for finding roots of nonlinear equations.
    This method uses a modification to the standard Newton's method that includes
    second derivative information to accelerate convergence. It is particularly
    useful for multiple roots where the standard Newton's method might converge slowly.

    Parameters
    ----------
    f : Callable[[float], float]
        The function for which we want to find the root.
    x0 : float
        Initial guess for the root.
    delta : float
        Tolerance for the change in x between iterations. Default is 1e-6.
    epsilon : float
        Tolerance for the function value at the approximate root. Default is 1e-6.
    m : int
        Maximum number of iterations. Default is 100.

    Returns
    -------
    Tuple[float, float, int, float]
        - Approximation of the root
        - Absolute error estimate
        - Number of iterations performed
        - Function value at the approximated root

    Raises
    ------
    ZeroDivisionError
        If the denominator in the iteration formula becomes zero.
    ValueError
        If the method fails to converge within the specified number of iterations.

    Notes
    -----
    The formula used is:
        x_{n+1} = x_n - (f(x_n) * f'(x_n)) / (f'(x_n)^2 - f(x_n) * f''(x_n))
    This method generally has cubic convergence for simple roots and
    quadratic convergence for multiple roots, compared to the linear
    convergence of standard Newton's method for multiple roots.
    """
    for i in range(m + 1):
        df = _numerical_derivative(f, x0)
        d2f = _numerical_second_derivative(f, x0)

        denominator = df**2 - f(x0) * d2f
        if denominator == 0:  # Prevent division by zero
            raise ZeroDivisionError("Division by zero")

        p = x0 - (f(x0) * df) / denominator
        e = np.abs(p - x0)
        x0 = p
        y = f(p)

        if (np.abs(y) < epsilon) or (e < delta):
            return p, e, i + 1, y

    raise ValueError("Method failed after {m} iterations")


def f(n):

    return 1/3 * (n**2 / (n**2 - (1.552)**2) + n**2 / (n**2 - (1.582)**2) +n**2 / (n**2 - (1.588)**2) - 1)


n = np.linspace(1.584,1.587,num=1000)

plt.plot(n,f(n))
plt.grid()
print(newtonMod(f,1.584))
# print(newtonMod(f,1.58))
# print(newtonMod(f,1.589))
plt.show()
