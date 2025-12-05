"""
Se definiran aqui las funciones que se necesiten para el trabajo para ser luego usadas en en la terminal principal
"""

# Librerias a usar
import numpy as np
import sympy as sp

import sympy.vector as vec
from sympy.vector import CoordSys3D

from IPython.display import display, Math
import scipy.integrate as integ
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Se definiran los parametros iniciales

u0r = 1
u0 = 4 * math.pi * 10e-7
e0r = 1
e0 = 8.854e-12
p = 10e-9
q = 2 * math.pi / p
c = 299792458

## Parametros de entrada
delta_ab = 0.001
epsilon_ab = 1.41 ** 2
lambda0 = 500E-9
k0 = 2 * math.pi / lambda0
alf = lambda0 / p

nx = math.sqrt(epsilon_ab + delta_ab)
ny = math.sqrt(epsilon_ab - delta_ab)

r = CoordSys3D('r')
z, t = sp.symbols('z t')


# Con esta funcion definimos los autovalores dependientes del valor de lambda
def N_(lam, type, dir):
    if type == 1 and dir == 1:
        auto_vals = + ((lam / p) ** 2 + epsilon_ab +
                       math.sqrt(4 * (lam / p) ** 2 * epsilon_ab + delta_ab ** 2)) ** (0.5) + 1j * 0
    elif type == 1 and dir == -1:
        auto_vals = - ((lam / p) ** 2 + epsilon_ab +
                       math.sqrt(4 * (lam / p) ** 2 * epsilon_ab + delta_ab ** 2)) ** (0.5) + 1j * 0
    elif type == 2 and dir == 1:
        auto_vals = + ((lam / p) ** 2 + epsilon_ab -
                       math.sqrt(4 * (lam / p) ** 2 * epsilon_ab + delta_ab ** 2)) ** (0.5) + 1j * 0
    elif type == 2 and dir == -1:
        auto_vals = - ((lam / p) ** 2 + epsilon_ab -
                       math.sqrt(4 * (lam / p) ** 2 * epsilon_ab + delta_ab ** 2)) ** (0.5) + 1j * 0
    return auto_vals


# Definimos con esta funcion los autovectores representativos como base
# dependientes del valor que adopte n como entrada
def Aprime_0(n):
    A0_vec = ((1 * r.i + -sp.I * (nx ** 2 - n ** 2 - (alf) ** 2) / (2 * alf * n) * r.j)
              * sp.exp(-sp.I * (z * k0 * n)))

    return A0_vec


# Definición en forma simbólica de la forma incial de la onda electromagnética
# incidente

# def Evec(Ax, Ay, t, omega, dir):
#     temporal_phase = sp.exp(sp.print(1 - fun.find_amplitudes_LCP_RCP(lam, nh, h, p, eav, del_av, "RCP"))
#     I * (omega * t))
#     if dir == +1:
#         E0_vec = (Ax * r.i + Ay * r.j) * sp.exp(-sp.I * (z * k0))
#     elif dir == -1:
#         E0_vec = (Ax * r.i + Ay * r.j) * sp.exp(sp.I * (z * k0))
#     return E0_vec * temporal_phase


# Definimos una función que halle las amplitudes de entrada con respecto a las amplitudes de los
# Eigenmodos

def find_amplitudes_LCP_RCP(spectrum, n_medium, width, pitch, eav, del_av, pol):

    t, ncnc1, lam = sp.symbols('t ncnc1 lam', complex = True)
    u, r, v1, v2, w, ncnc2 = sp.symbols('u r v1 v2 w ncnc2', complex = True)

    n_cnc1 = sp.sqrt(
        (lam / pitch) ** 2 + eav + sp.sqrt(del_av ** 2 + 4 * eav * (lam / pitch) ** 2 + sp.I * 0) + sp.I * 0)

    n_cnc2_2 = (lam / pitch) ** 2 + eav - sp.sqrt(del_av ** 2 + 4 * eav * (lam / pitch) ** 2)

    f1 = sp.Eq(u + r, v1 + v2)
    f2 = sp.Eq(u - r, -sp.I * w * v1 + sp.I * w * v2)

    sol1 = sp.solve((f1, f2), (u, r))

    Reflection = np.array([])

    if pol == "LCP":

        alfa = spectrum/pitch

        W = sp.I * ((eav + del_av) - ncnc2 ** 2 - (lam / pitch) ** 2) / (2 * (lam / pitch) * ncnc2)

        f3 = sp.Eq(v1 * sp.exp(-sp.I * (2 * sp.pi / lam) * ncnc2 * width) +
                   v2 * sp.exp(sp.I * (2 * sp.pi / lam) * ncnc2 * width)
                   , t * sp.exp(-sp.I * (2 * sp.pi / lam) * n_medium * width))

        f4 = sp.Eq(v1 * w * sp.exp(-sp.I * (2 * sp.pi / lam) * ncnc2 * width) -
                   v2 * w * sp.exp(sp.I * (2 * sp.pi / lam) * ncnc2 * width)
                   , sp.I * t * sp.exp(-sp.I * (2 * sp.pi / lam) * n_medium * width))

        sol2 = sp.solve((f3, f4), (v1, v2))

        sol2[v1] = sp.simplify(sol2[v1])
        sol2[v2] = sp.simplify(sol2[v2])

        R = (sol1[r] / sol1[u])

        R = sp.simplify(R)

        R = R.subs({v1: sol2[v1], v2: sol2[v2]})

        R = sp.simplify(R)## Todo esta correcto hasta aca

        ncnc_num = sp.lambdify(lam, n_cnc2_2, "numpy"); ncnc_arr = np.sqrt(ncnc_num(spectrum) + 1j*0)
        n2 = np.sqrt(alfa ** 2 + eav - np.sqrt(del_av ** 2 + 4 * eav * alfa ** 2 + 0j) + 0j)
        w_num = sp.lambdify((lam,ncnc2), W, "numpy"); w_arr = w_num(spectrum, ncnc_arr)

        R = sp.lambdify((lam, w, ncnc2), R, "numpy")

        Reflection = R(spectrum, w_arr, ncnc_arr)

    elif pol == "RCP":

        W = sp.I * ((epsilon_ab + delta_ab) - n_cnc1 ** 2 - (lam / pitch) ** 2) / (2 * (lam / pitch) * n_cnc1)

        f3 = sp.Eq(v1 * sp.exp(sp.simplify(-sp.I * (2 * sp.pi / lam) * ncnc1 * width)) + v2 * sp.exp(
            sp.simplify(sp.I * (2 * sp.pi / lam) * ncnc1 * width))
                   , t * sp.exp(sp.simplify(-sp.I * (2 * sp.pi / lam) * n_medium * width)))

        f4 = sp.Eq(v1 * w * sp.exp(sp.simplify(-sp.I * (2 * sp.pi / lam) * ncnc1 * width)) - v2 * w * sp.exp(
            sp.simplify(sp.I * (2 * sp.pi / lam) * ncnc1 * width))
                   , sp.I * t * sp.exp(sp.simplify(-sp.I * (2 * sp.pi / lam) * n_medium * width)))

        sol2 = sp.solve((f3, f4), (v1, v2))

        sol2[v1] = sp.simplify(sol2[v1])
        sol2[v2] = sp.simplify(sol2[v2])

        R = (sol1[r] / sol1[u])

        R = sp.simplify(R.subs({v1: sol2[v1], v2: sol2[v2]}))


        w_num = sp.lambdify((lam),W, "numpy"); w_arr = w_num (spectrum)
        ncnc_num = sp.lambdify((lam),n_cnc1, "numpy"); ncnc_arr = ncnc_num(spectrum)

        R = sp.lambdify((lam,w,ncnc1), R, "numpy")

        Reflection = R(spectrum,w_arr,ncnc_arr)

    R1 = (np.abs(Reflection))**2

    return R1


# Definimos una funcion que halle las amplitudes de etrada con respecto a las aplitudes
# de los eigenmodos, y con la luz de salida. para luz de entrada linealmente polarizada.

def find_amplitudes_LinPol(spectrum, n_medium, theta, width, pitch, eav, del_av):

    t1, ncnc1, lam = sp.symbols('t ncnc1 lam')
    u, r, v1, v2, v3, v4, w, ncnc2 = sp.symbols('u r v1 v2 w ncnc2')

    n_cnc1 = sp.sqrt(
        (lam / pitch) ** 2 + eav + sp.sqrt(del_av ** 2 + 4 * eav * (lam / pitch) ** 2 + sp.I * 0) + sp.I * 0)
    n_cnc2_2 = (lam / pitch) ** 2 + eav - sp.sqrt(del_av ** 2 + 4 * eav * (lam / pitch) ** 2)

    w1 = sp.I * ((eav + del_av) - ncnc1 ** 2 - (lam/pitch) ** 2) / (2 * (lam/pitch) * ncnc1)
    w2 = sp.I * ((eav + del_av) - ncnc2 ** 2 - (lam/pitch) ** 2) / (2 * (lam/pitch) * ncnc2)

    t2 = u / sp.sqrt(2) * sp.exp(-sp.I * theta)

    f3 = sp.Eq(v1 * sp.exp(-sp.I * k0 * ncnc2 * width) + v2 * sp.exp(sp.I * k0 * ncnc2 * width) +
               v3 * sp.exp(-sp.I * k0 * ncnc1 * width) + v4 * sp.exp(sp.I * k0 * ncnc1 * width)
               , (t1 + t2) * sp.exp(-sp.I * k0 * n_medium * width))

    f4 = sp.Eq(-w2 * v1 * sp.exp(-sp.I * k0 * ncnc2 * width) + w2 * v2 * sp.exp(sp.I * k0 * ncnc2 * width) +
               w1 * v3 * sp.exp(-sp.I * k0 * ncnc1 * width) - w1 * v4 * sp.exp(sp.I * k0 * ncnc1 * width)
               , (-t1 + t2) * sp.I * sp.exp(-sp.I * k0 * n_medium * width))

    f5 = sp.Eq(-ncnc2 * v1 * sp.exp(-sp.I * k0 * ncnc2 * width) + ncnc2 * v2 * sp.exp(sp.I * k0 * ncnc2 * width) -
               ncnc1 * v3 * sp.exp(-sp.I * k0 * ncnc1 * width) + ncnc1 * v4 * sp.exp(sp.I * k0 * ncnc1 * width)
               , -(t1 + t2) * sp.I * k0 * n_medium * sp.exp(-sp.I * k0 * n_medium * width))

    f6 = sp.Eq(
        ncnc2 * w2 * v1 * sp.exp(-sp.I * k0 * ncnc2 * width) + ncnc2 * w2 * v2 * sp.exp(sp.I * k0 * ncnc2 * width) -
        ncnc1 * w1 * v3 * sp.exp(-sp.I * k0 * ncnc1 * width) - ncnc1 * w1 * v4 * sp.exp(sp.I * k0 * ncnc1 * width)
        , (-t1 + t2) * k0 * n_medium * sp.exp(-sp.I * k0 * n_medium * width))

    sol2 = sp.solve((f3, f4, f5, f6), (v1, v2, v3, v4))

    f1 = sp.Eq(u * sp.cos(theta) + r * sp.cos(theta),
               1 / math.sqrt(2) * (sol2[v1] + sol2[v2] + sol2[v3] + sol2[v4]))
    f2 = sp.Eq(u * sp.sin(theta) - r * sp.sin(theta),
               1 / math.sqrt(2) * (-w2 * sol2[v1] + w2 * sol2[v2] + w1 * sol2[v3] - w1 * sol2[v4])) # Pendiente para probar
                                                                                                    #hasta aca

    sol1 = sp.solve((f1, f2), (u, r))

    R = sp.simplify((sp.conjugate(sol1[r] / sol1[u]) * (sol1[r] / sol1[u])) ** 2)

    R = sp.lambdify((lam), R, "numpy")

    Reflection = R(spectrum)

    return Reflection


# Definición en forma simbólica del operador rotacion S(theta)

def S(phi):
    # al ser el input negativo obtenemos S-1,no necesitamos definirlo aparte

    matr = sp.Matrix([[sp.cos(phi), sp.sin(phi)],
                      [-sp.sin(phi), sp.cos(phi)]])
    return matr


def Matx_vec_mult(M, V):
    M = sp.sympify(M)
    V = sp.sympify(V)

    vec_matr = sp.Matrix(V.to_matrix(r))

    mult = M * vec_matr
    re_vec = mult[0] * r.i + mult[1] * r.j + mult[2] * r.k

    return re_vec



