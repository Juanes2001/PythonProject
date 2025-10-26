import numpy as np
import sympy as sp
import functions as fn
from IPython.display import display, Math
import scipy.integrate as int
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

hbar = 1
me   = 1
mp = 1836.15
u = (me*mp)/(me+mp)
Z = 1

w, r, phi, theta , phi_p,theta_p = sp.symbols('w r phi theta phi_p theta_p', real=True)
f1 = sp.Function('R_{nl}')(phi,theta)
f2 = sp.Function('R_{nl}')(phi_p,theta_p)
hb = sp.symbols('hbar')
N,L,M = sp.symbols('n l m', integer=True)

"""
Como principalmente para este taller se conocerán las transiciones energeticas y las series medidas, se necesitan las conversiones de unidades atomicas a unidades energeticas, y tambien unidades de longitud de onda.
"""

hart2cm = 455.633 # en angstroms
hart2ev = 27.2116 # en electronvoltios


# Ahora definamos las funciones radiales y su densidad, es decir, el módulo de estas funciones al cuadrado. Lo haremos de forma similas a como hicimos con los armonicos esfericos

def eferic_armonics_symbol(var1,var2,l,m):
    ef = (sp.sqrt((2*l+1)/(4*sp.pi) * sp.factorial(l-abs(m))/sp.factorial(l+abs(m)) )
          * sp.exp(sp.I * m * var2) * 1/(2**l * sp.factorial(l)) *
          (1-w**2)**(abs(m)/2)* sp.Derivative((w**2-1)**l,(w,l+abs(m))))
    ef = ef.subs(w,sp.cos(var1))
    return ef

def eferic_armonics_numeric(var1,var2,l,m):
    ef = (sp.sqrt((2*l+1)/(4*sp.pi) * sp.factorial(l-abs(m))/sp.factorial(l+abs(m)) )
          * sp.exp(sp.I * m * var2) * 1/(2**l * sp.factorial(l)) *
          (1-w**2)**(abs(m)/2)* sp.diff((w**2-1)**l,(w,l+abs(m))))
    ef = ef.subs(w,sp.cos(var1))
    return ef

def Radial_part_numeric(var1,n,l):
    p = 2*Z*u/n  * var1
    radial = sp.S(0)
    for k in range(n-l):
        radial +=  (-1)**(k+1) * (sp.factorial(n+l))**2/ (sp.factorial(n-l-1-k)*sp.factorial(2*l+1+k)*sp.factorial(k)) * p**k

    radial *= - sp.sqrt(((2*Z*u/n)**3 * sp.factorial(n-l-1)/(2*n*(sp.factorial(n+l))**3  ))) * sp.exp(-p/2) * p**l

    return radial

def integrate_2d(Z, x, y, method):
    """
    Z: 2D array shaped (len(y), len(x))  -> f(y_i, x_j)
    x: 1D array (len Nx)
    y: 1D array (len Ny)
    method: 'rect', 'trapz', or 'simps'
    """
    if method == 'rect':
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        return np.sum(Z) * dx * dy
    elif method == 'trapz':
        return np.trapezoid(np.trapezoid(Z, x=x, axis=1), x=y, axis=0)
    elif method == 'simps':
        return int.simpson(int.simpson(Z, x=x, axis=1), x=y, axis=0)
    else:
        raise ValueError("method must be 'rect','trapz' or 'simps'")


def safe_lambdify(vars, expr):
    f = sp.lambdify(vars, expr, "numpy")
    return lambda *args: np.broadcast_to(f(*args), np.broadcast(*args).shape)

thetvals1,thetvals2,phivals = np.linspace(0,np.pi/2,250),np.linspace(np.pi/2,np.pi,250) , np.linspace(0,2*np.pi,500)
Tvals1,Pvals = np.meshgrid(thetvals1,phivals) # x, y
Tvals2,Pvals = np.meshgrid(thetvals2,phivals) # x, y

esferic_num =safe_lambdify((theta,phi), eferic_armonics_numeric(theta,phi,1,0))
rr1 = np.abs(esferic_num(Tvals1,Pvals))
rr2 = np.abs(esferic_num(Tvals2,Pvals))


X1 = rr1*np.sin(Tvals1)*np.cos(Pvals)
Y1 = rr1*np.sin(Tvals1)*np.sin(Pvals)
Z1 = rr1*np.cos(Tvals1)

X2 = rr2*np.sin(Tvals2)*np.cos(Pvals)
Y2 = rr2*np.sin(Tvals2)*np.sin(Pvals)
Z2 = rr2*np.cos(Tvals2)

X = np.concatenate([X1, X2], axis=0)
Y = np.concatenate([Y1, Y2], axis=0)
Z = np.concatenate([Z1, Z2], axis=0)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)
ax.set_xlim(-0.4,0.4)
ax.set_ylim(-0.4,0.4)
ax.set_zlim(-0.4,0.4)

fig.show()

