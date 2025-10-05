"""
En este programa programaremos para varios paquetes de ondas desarrolladas en la base DVR
la propagacion de una particula para dos casos, particula libre y particula encerrada en un pozo de paredes infinitas

Por lo que primero buscamos la manera que desde una funcion analitica cuadrado integrable crear una funcion que la desarrolle
automaticamente en la base de DVR, segundo, se define el Hamiltoniano para cada uno de los casos y asi luego se diagonaliza
para hallar los autoestados de energia, Y con ello los autovectores que coincide con los coeficientes de la base DVR para cada autoestado

Por ultimo y usando el propagador temporal, se hallan los coeficientes Cn para desarrollar Phi(x,t) en la base de autoestados, y luego
se hace la animación

"""
import math

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functions as fn
L = 20
Nf = 300
x0 = 10
t0 = 0
p0 = 1

hbar = 1
me = 1
sigma = 0.6

# Definimos las funciones DVR con numpy

x,t= sp.symbols('x t', real = True)
n,k,ii,jj= sp.symbols('n k ii jj', integer = True)

f = 2/sp.sqrt(L*(Nf+1)) * sp.summation(sp.sin((k*sp.pi*x)/L)*sp.sin((k*n*sp.pi)/(Nf+1)), (k,1,Nf) )

fnum = sp.lambdify((x,n), f, modules='numpy')

# Veamos la representación de la función rectángulo en esta base

xvals = np.linspace(0,20,Nf)


def rect(x):
    return np.where(np.abs(x-x0) <= 1/2, np.exp(1j*p0*x), 0)

an = []
dn = []
for i in range(1,Nf+1):
    an.append(fn.dot(xvals,fnum(xvals,i)/fn.dot(xvals,fnum(xvals,i),fnum(xvals,i)),rect(xvals)))
    dn.append(math.sqrt(L/(Nf+1))*rect(i*L/(Nf+1)))
an = np.array(an)
dn = np.array(dn)

phi_0_RealConsts =np.zeros(Nf)
phi_0_NonRealConsts =np.zeros(Nf)

for i in range(0,Nf):
    phi_0_RealConsts    = phi_0_RealConsts    + dn[i]*fnum(xvals,i)
    phi_0_NonRealConsts = phi_0_NonRealConsts + an[i]*fnum(xvals,i)

fig1,ax1 = plt.subplots()
ax1.set_xlim(0,20)
ax1.set_ylim(-1,2)

ax1.plot(xvals,np.abs(rect(xvals))+2)
ax1.plot(xvals,np.abs(phi_0_RealConsts)+1)
ax1.plot(xvals,np.abs(phi_0_NonRealConsts))

plt.grid()

# Ahora tendremos que usar esta base para hallar la matriz del Hamiltoniano. Para ello aplicamos <fi|H|fj>, que para el caso
# de una particula libre se trata solo de la solucion de la integral de fi*ddfj, la cual ya le tenemos una solucion analitica

fiddfj =  2*(sp.pi)**2/((Nf+1)*L**2) * sp.summation(k**2 * sp.sin( ii*k*sp.pi/(Nf+1) )*sp.sin( jj*k*sp.pi/(Nf+1) ),(k,1,Nf) )
fiddfj_num = sp.lambdify((ii,jj), fiddfj, modules='numpy')

# asi entonces definimos las entradas de nuestra matriz

H = np.zeros((Nf,Nf))

for l in range(Nf):
    for m in range(Nf):
        H[l,m] = -1/2 * fiddfj_num(l+1,m+1)

# teniendo el Hamiltiniano, Diagonalizamos para obtener los autovalores

En, bn = np.linalg.eig(H)

# De este resultado buscamos llegar a Psi(x,t) = sum cn phi_n(x) * exp(-i*En*t) en donde los cn se hallan como
# cn = sum(bn^i* * ai)


cn = []

for l in range(Nf):
    bn[:, l:l + 1] = bn[:, l:l + 1] / math.sqrt(sum((bn[:, l:l + 1].T)[0] ** 2))
    cn.append(sum((bn[:,l:l+1].T)[0]*an))


# Formamos los autoestados en forma de función

cn = np.array(cn)
cn = cn/np.linalg.norm(cn)

phin = np.zeros((Nf,Nf))
for l in range(Nf):
    phin[l:l+1,:] = np.array((bn[l:l+1,:])[0]*fnum(xvals,l+1)/fn.dot(xvals,fnum(xvals,l+1),fnum(xvals,l+1)))

# Con esto ya podemos desarrollar la propagacion temporal:

tt = np.linspace(0,3,Nf)

def psi(tt):
    Psi = np.zeros(Nf)
    for l in range(Nf):
        Psi = Psi + cn[l]*(phin[:,l:l+1].T)[0]*np.exp(-1j*En[l]*(tt-t0))
    return Psi

fig2,ax2 = plt.subplots()
ax2.set_xlim(0, 20)
ax2.set_ylim(-1, 4)
line_module, = ax2.plot([], [], lw=2, color = 'blue')
frame_text = ax2.text(0.02, 0.9, '', transform=ax2.transAxes, fontsize=12, color='red')

def init():
    line_module.set_data([],[])
    frame_text.set_text('')
    return line_module, frame_text

def animate(frames):
    y_vals = psi(tt[frames])  # arguments: (x,t,p0,x0,t0)
    line_module.set_data(xvals, np.abs(y_vals))  # real part
    frame_text.set_text(f"{frames}")
    return line_module,frame_text


ani = animation.FuncAnimation(fig2, animate,init_func=init, frames=Nf, interval=50, blit=True)


ax2.plot(xvals,phin[:,0:1])
print(phin)

plt.grid()
plt.show()

