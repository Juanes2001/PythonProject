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

    t1, ncnc1, lam = sp.symbols('t1 ncnc1 lam')
    u, r, v1_alfa, v2_alfa, v3_alfa, v4_alfa, w1, w2, ncnc2 = sp.symbols('u r v1_alfa v2_alfa v3_alfa v4_alfa w1 w2 ncnc2')

    v1_beta, v2_beta, v3_beta, v4_beta = sp.symbols('v1_beta v2_beta v3_beta v4_beta')

    A,Ap,B,Bp,C,Cp,K0 = sp.symbols('A Ap B Bp C Cp K0')

    k0 = 2 * sp.pi / lam

    A_form = sp.exp(-sp.I * k0 * ncnc2 * width)
    Ap_form = sp.conjugate(sp.exp(-sp.I * k0 * ncnc2 * width))
    B_form = sp.exp(-sp.I * k0 * ncnc1 * width)
    Bp_form = sp.conjugate(sp.exp(-sp.I * k0 * ncnc1 * width))
    C_form = sp.exp(-sp.I * k0 * n_medium * width)
    Cp_form = sp.exp(-sp.I*(k0*n_medium*width + theta))


    n_cnc1 = sp.sqrt(
        (lam / pitch) ** 2 + eav + sp.sqrt(del_av ** 2 + 4 * eav * (lam / pitch) ** 2 + sp.I * 0) + sp.I * 0)
    n_cnc2_2 = (lam / pitch) ** 2 + eav - sp.sqrt(del_av ** 2 + 4 * eav * (lam / pitch) ** 2)

    W1 = sp.I * ((eav + del_av) - ncnc1 ** 2 - (lam/pitch) ** 2) / (2 * (lam/pitch) * ncnc1)
    W2 = sp.I * ((eav + del_av) - ncnc2 ** 2 - (lam/pitch) ** 2) / (2 * (lam/pitch) * ncnc2)

    t2 = u / sp.sqrt(2) * sp.exp(-sp.I * theta)

    f1_alfa = sp.Eq(sp.simplify(v1_alfa * A + v2_alfa * Ap + v3_alfa * B + v4_alfa * Bp),                                                          t1 * C)
    f2_alfa = sp.Eq(sp.simplify(-w2 * v1_alfa * A + w2 * v2_alfa * Ap +w1 * v3_alfa * B - w1 * v4_alfa * Bp),                                     -t1 * C* sp.I )
    f3_alfa = sp.Eq(sp.simplify(-ncnc2 * v1_alfa * A + ncnc2 * v2_alfa * Ap -ncnc1 * v3_alfa * B + ncnc1 * v4_alfa * Bp),                         -t1 * sp.I * K0 * n_medium * C)
    f4_alfa = sp.Eq(sp.simplify(ncnc2 * w2 * v1_alfa * A + ncnc2 * w2 * v2_alfa * Ap -ncnc1 * w1 * v3_alfa * B - ncnc1 * w1 * v4_alfa * Bp),      -t1 * K0 * n_medium * C)

    V_alfa = sp.solve((f1_alfa, f2_alfa, f3_alfa, f4_alfa), (v1_alfa, v2_alfa, v3_alfa, v4_alfa))


    a1 = V_alfa[v1_alfa] = sp.simplify(V_alfa[v1_alfa])/t1; a1 = a1.subs({A: A_form,C: C_form, K0: k0, w1: W1.subs({ncnc1: n_cnc1}),ncnc1: n_cnc1, w2: W2})
    a2 = V_alfa[v2_alfa] = sp.simplify(V_alfa[v2_alfa])/t1; a2 = a2.subs({Ap: Ap_form,C: C_form, K0: k0, w1: W1.subs({ncnc1: n_cnc1}),ncnc1: n_cnc1, w2: W2})
    a3 = V_alfa[v3_alfa] = sp.simplify(V_alfa[v3_alfa])/t1; a3 = a3.subs({B: B_form,C: C_form, K0: k0, w1: W1.subs({ncnc1: n_cnc1}),ncnc1: n_cnc1, w2: W2})
    a4 = V_alfa[v4_alfa] = sp.simplify(V_alfa[v4_alfa])/t1; a4 = a4.subs({Bp: Bp_form,C: C_form, K0: k0, w1: W1.subs({ncnc1: n_cnc1}),ncnc1: n_cnc1, w2: W2})

    f1_beta= sp.Eq(sp.simplify(v1_beta * A + v2_beta * Ap + v3_beta * B + v4_beta * Bp),
                    u/math.sqrt(2) * Cp)
    f2_beta = sp.Eq(sp.simplify(-w2 * v1_beta * A + w2 * v2_beta * Ap + w1 * v3_beta * B - w1 * v4_beta * Bp),
                    u/math.sqrt(2) * Cp * sp.I)
    f3_beta = sp.Eq(sp.simplify(-ncnc2 * v1_beta * A + ncnc2 * v2_beta * Ap - ncnc1 * v3_beta * B + ncnc1 * v4_beta * Bp),
                   -u/math.sqrt(2) * sp.I * K0 * n_medium * Cp)
    f4_beta = sp.Eq(sp.simplify(ncnc2 * w2 * v1_beta * A + ncnc2 * w2 * v2_beta * Ap - ncnc1 * w1 * v3_beta * B - ncnc1 * w1 * v4_beta * Bp),
                    u/math.sqrt(2) * K0 * n_medium * Cp)

    V_beta = sp.solve((f1_beta, f2_beta, f3_beta, f4_beta), (v1_beta, v2_beta, v3_beta, v4_beta))

    b1 = V_beta[v1_beta] = sp.simplify(V_beta[v1_beta])/u; b1 = b1.subs({A: A_form,Cp: Cp_form, K0: k0, w1: W1.subs({ncnc1: n_cnc1}),ncnc1: n_cnc1, w2: W2})
    b2 = V_beta[v2_beta] = sp.simplify(V_beta[v2_beta])/u; b2 = b2.subs({Ap: Ap_form,Cp: Cp_form, K0: k0, w1: W1.subs({ncnc1: n_cnc1}),ncnc1: n_cnc1, w2: W2})
    b3 = V_beta[v3_beta] = sp.simplify(V_beta[v3_beta])/u; b3 = b3.subs({B: B_form,Cp: Cp_form, K0: k0, w1: W1.subs({ncnc1: n_cnc1}),ncnc1: n_cnc1, w2: W2})
    b4 = V_beta[v4_beta] = sp.simplify(V_beta[v4_beta])/u; b4 = b4.subs({Bp: Bp_form,Cp: Cp_form, K0: k0, w1: W1.subs({ncnc1: n_cnc1}),ncnc1: n_cnc1, w2: W2})

    G = sp.sec(theta) - w2*sp.csc(theta); G = G.subs({w2: W2})
    Gp = sp.sec(theta) + w2*sp.csc(theta); Gp = Gp.subs({w2: W2})
    H = sp.sec(theta) - w1 * sp.csc(theta); H = H.subs({w1: W1.subs({ncnc1: n_cnc1})})
    Hp = sp.sec(theta) + w1 * sp.csc(theta); Hp = Hp.subs({w1: W1.subs({ncnc1: n_cnc1})})

    sq2 = math.sqrt(2)

    t_u = (2*sq2 - b1*G/sq2 - b2*Gp/sq2 - b3*Hp/sq2 - b4*H/sq2)/(G*a1 + Gp*a2 + Hp*a3 + H*a4)

    R = (Gp*(t_u*a1+b1/sq2) + G*(t_u*a2+b2/sq2) + H*(t_u*a3+b3/sq2) + Hp*(t_u*a4+b4/sq2))/(2*sq2)

    ncnc2_num = sp.lambdify((lam), n_cnc2_2, "numpy");ncnc2_arr = np.sqrt(ncnc2_num(spectrum) + 1j*0)

    R = sp.lambdify((lam,ncnc2), R, "numpy")

    Reflection = R(spectrum, ncnc2_arr)

    R1 = (np.abs(Reflection)) ** 2

    return R1


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



