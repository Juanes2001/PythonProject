import numpy as np
import sympy as sp
import functions as fn
from IPython.display import display, Math
import scipy.integrate as int


"""
Definición de operadores para dos sistenas de referencia distintos, y una base conjunta que pueda ser autofunción de tales operadores
"""

#Definimos parametros a usar en unidades atómicas
hbar = 1
me   = 1

w, phi, theta , phi_p,theta_p = sp.symbols('w phi theta phi_p theta_p', real=True)
f1 = sp.Function('Y_{lm}')(phi,theta)
f2 = sp.Function('Y_{lm}')(phi_p,theta_p)
hb = sp.symbols('hbar')
L,M = sp.symbols('l m', integer=True)


# Ahora vamos a comprobar su funcionalidad aplicando sobre ellos los armonicos esfericos.

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


display (Math(r"Y_{lm}(\theta,\phi) =" + sp.latex(eferic_armonics_symbol(theta,phi,L,M))))
display (Math(r"Y_{lm}(\theta_{p},\phi_{p}) =" + sp.latex(eferic_armonics_symbol(theta_p,phi_p,L,M))))

# Definimos la multiplicación de los armonicos esfericos para cada marco de coordenadasç



def Y1_Y2  (l,m,lp,mp):
    mult_esferic = eferic_armonics_numeric(theta,phi,l,m) * eferic_armonics_numeric(theta_p,phi_p,lp,mp)
    return mult_esferic


def Lz_op (var1,var2,f):
    Lz = -sp.I*hb*sp.diff(f,var2)
    return Lz

def Lx_op (var1,var2,f):
    Lx = sp.I*hb*(-sp.sin(var2)*sp.diff(f,var1) + sp.cot(var1)*sp.cos(var2)*sp.diff(f,var2))
    return Lx

def Ly_op (var1,var2,f):
    Ly = -sp.I*hb*(sp.cos(var2)*sp.diff(f,var1)   -  sp.cot(var1)*sp.sin(var2)*sp.diff(f,var2))
    return Ly

# Ahora sigamos con el operador L2

def L2_op (var1,var2,f):
    L2 = -hb**2*(sp.diff(f,var1,2)   +  sp.cot(var1)*sp.diff(f,var1)+1/(sp.sin(var1))**2 *sp.diff(f,var2,2))
    return L2


# Ya teniendo los operadores L^2_1, L^2_2, Lz_1, Lz_2 , aplicado sobre el producto de los armónicos esféricos para cada coordenada

# Primero para L2 se tiene que L2(Ylm) = hbar *l*(l+1)Ylm

for l in range(2):
    for m in range(-l,l+1):
        display (Math(r"\hat{L^{2}_{1}}"+f"(Y^1_{{{l}{m}}}"+r"*Y^{2}_{l_{p}m_{p}}) =" + sp.latex(sp.trigsimp(sp.nsimplify(sp.simplify(
                                                                        sp.together(
                                                                            sp.trigsimp(
                                                                                sp.powsimp(
                                                                                    sp.cancel(
                                                                                        sp.factor(
                                                                                            sp.expand(L2_op(theta,phi,Y1_Y2(l,m,L,M)
                                                                                                      )/Y1_Y2(l,m,L,M))
                                                                                        )
                                                                                    ), force=True
                                                                                )
                                                                            )
                                                                        )
                                                                    ) , rational=True, tolerance=1e-12) )  )))



print("\n")


for l in range(2):
    for m in range(-l,l+1):
        display (Math(r"\hat{L^{2}_{2}}"+f"(Y^2_{{{l}{m}}}"+r"*Y^{1}_{l_{p}m_{p}}) =" + sp.latex(sp.trigsimp(sp.nsimplify(sp.simplify(
                                                                        sp.together(
                                                                            sp.trigsimp(
                                                                                sp.powsimp(
                                                                                    sp.cancel(
                                                                                        sp.factor(
                                                                                            sp.expand(L2_op(theta_p,phi_p,Y1_Y2(L,M,l,m)
                                                                                                      )/Y1_Y2(L,M,l,m))
                                                                                        )
                                                                                    ), force=True
                                                                                )
                                                                            )
                                                                        )
                                                                    ) , rational=True, tolerance=1e-12) )  )))



# Hacemos lo mismo para Lz_1 y Lz_2 se tiene que Lz_i(Ylm) = hbar *l*(l+1)Ylm

for l in range(2):
    for m in range(-l,l+1):
        display (Math(r"\hat{L_{z1}}"+f"(Y^1_{{{l}{m}}}"+r"*Y^{2}_{l_{p}m_{p}}) =" + sp.latex(sp.trigsimp(sp.nsimplify(sp.simplify(
                                                                        sp.together(
                                                                            sp.trigsimp(
                                                                                sp.powsimp(
                                                                                    sp.cancel(
                                                                                        sp.factor(
                                                                                            sp.expand(Lz_op(theta,phi,Y1_Y2(l,m,L,M)
                                                                                                      )/Y1_Y2(l,m,L,M))
                                                                                        )
                                                                                    ), force=True
                                                                                )
                                                                            )
                                                                        )
                                                                    ) , rational=True, tolerance=1e-12) )  )))



print("\n")


for l in range(2):
    for m in range(-l,l+1):
        display (Math(r"\hat{L_{z2}}"+f"(Y^2_{{{l}{m}}}"+r"*Y^{1}_{l_{p}m_{p}}) =" + sp.latex(sp.trigsimp(sp.nsimplify(sp.simplify(
                                                                        sp.together(
                                                                            sp.trigsimp(
                                                                                sp.powsimp(
                                                                                    sp.cancel(
                                                                                        sp.factor(
                                                                                            sp.expand(Lz_op(theta_p,phi_p,Y1_Y2(L,M,l,m)
                                                                                                      )/Y1_Y2(L,M,l,m))
                                                                                        )
                                                                                    ), force=True
                                                                                )
                                                                            )
                                                                        )
                                                                    ) , rational=True, tolerance=1e-12) )  ) ))

    # Este producto de Armonicos Esfericos no será autoestado de J^2  con J = L_1 +L_2

    # Definimos la funcion operador que incluye a estos dos operadores


def J2_op(f):
    j2 = L2_op(theta, phi, f) + L2_op(theta_p, phi_p, f) + 2 * (
                Lx_op(theta, phi, f) + Ly_op(theta, phi, f) + Lz_op(theta, phi, f)) * \
         (Lx_op(theta_p, phi_p, f) + Ly_op(theta_p, phi_p, f) + Lz_op(theta_p, phi_p, f))
    return j2


for l in range(2):
    for m in range(-l, l + 1):
        display(Math(r"\hat{J^{2}}" + f"(Y^1_{{{l}{m}}}*Y^2_{{{l}{m}}}) =" + sp.latex(
            J2_op(Y1_Y2(l, m, l, m)) / Y1_Y2(l, m, l, m))))

# Lo que sigue es construir las autofunciones para el momento aungular total, las cuales se contruyen usando los cooeficientes de Clebsed Gordan, por el cual vamos a definir una funcion que las calcule dependiendo de la autofuncion que se este construyendo

thetvals,phivals, thetvals_p,phivals_p = (np.linspace(0,np.pi,500) , np.linspace(0,2*np.pi,500),
                                          np.linspace(0,np.pi,500) , np.linspace(0,2*np.pi,500))
Tvals, Pvals = np.meshgrid(thetvals, phivals) # x, y
Tvals_p, Pvals_p = np.meshgrid(thetvals_p, phivals_p)
dtheta = sp.lambdify(theta,sp.sin(theta),"numpy")

def clebseth_G(j1,j2,m1,m2,j,m):

    esferic_num_sum = safe_lambdify((theta,phi,theta_p,phi_p), Y1_Y2(j1,m1,j2,m2))
    esferic_num_mult = safe_lambdify((theta,phi), eferic_armonics_numeric(theta,phi,j,m))

    return abs(integrate_2d(np.conjugate(esferic_num_sum(Tvals,Pvals,Tvals_p,Pvals_p))*
                            esferic_num_mult(Tvals,Pvals)*dtheta(Tvals),thetvals,phivals,'simps'))


#Con esto definimos entonces los nuevos autoestados

def sum_Y1_Y2(j1,j2,j,m):
    sum = sp.Integer(0)
    for m1 in range(-j1,j1+1):
        for m2 in range(-j2,j2+1):
            sum += sp.nsimplify(clebseth_G(j1,j2,m1,m2,j,m)*Y1_Y2(j1,m1,j2,m2), rational=True, tolerance=1e-15)
    return sum

# Con esto , comprobamos que para estos nuevos autoestados, si son autoestados de el momento angular total, para casos de j1 y j2 especificos, probaremos para varios casos, en donde j = |j1-j2|,...., j1+j2

for j1 in range(1):
    for j2 in range(2):
        for j in range(abs(j1-j2),j1+j2+1):
            for m in range(-j,j+1):
                display (Math(r"\hat{J^{2}}"+f"(Y^{{{j1}{j2}}}_{{{j}{m}}}) =" + sp.latex(sp.trigsimp(sp.simplify(
                                J2_op(sum_Y1_Y2(j1,j2,j,m))/sum_Y1_Y2(j1,j2,j,m)) )  )   ))
