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
Nf = 250
x0 = L/2-3
t0 = 0
p0 = 3

hbar = 1
me = 1
sigma = 0.4

# Definimos las funciones DVR con numpy



def f(x,n):
    fun_DVR = np.zeros(Nf)
    for k in range(Nf):
        fun_DVR = fun_DVR + (2/math.sqrt(L*(Nf+1)) * np.sin(((k+1)*math.pi*x)/L)*np.sin(((k+1)*n*math.pi)/(Nf+1)))
    return fun_DVR

# Veamos la representación de la función rectángulo en esta base

xvals = np.linspace(0,L,Nf)

def rect(x):

    gauss = 1/(math.sqrt(2*math.pi)*sigma) * np.exp(-(x-x0)**2/(2*sigma) + 1j*p0*x/hbar  )
    cuad = np.where(np.abs(x - x0) <= 1 / 2, np.exp(1j * p0 * x), 0)
    return gauss

an = []
dn = []

for i in range(1,Nf+1):
    an.append(fn.dot(xvals,f(xvals,i)/fn.dot(xvals,f(xvals,i),f(xvals,i)),rect(xvals)))
    dn.append(math.sqrt(L/(Nf+1))*rect(i*L/(Nf+1)))
an = np.array(an)
dn = np.array(dn)

phi_0_RealConsts =np.zeros(Nf)
phi_0_NonRealConsts =np.zeros(Nf)

for i in range(1,Nf+1):
    phi_0_RealConsts    = phi_0_RealConsts    + dn[i-1]*f(xvals,i)
    phi_0_NonRealConsts = phi_0_NonRealConsts + an[i-1]*f(xvals,i)

fig1,ax1 = plt.subplots()
ax1.set_xlim(0,L)
ax1.set_ylim(-1,4)

ax1.plot(xvals,np.abs(rect(xvals))+2)
ax1.plot(xvals,np.abs(phi_0_RealConsts)+1)
ax1.plot(xvals,np.abs(phi_0_NonRealConsts))

# Ahora tendremos que usar esta base para hallar la matriz del Hamiltoniano. Para ello aplicamos <fi|H|fj>, que para el caso
# de una particula libre se trata solo de la solucion de la integral de fi*ddfj, la cual ya le tenemos una solucion analitica
#

def  fiddfj(i,j):
    fiddfj_num = 0
    for k in range(Nf):
        fiddfj_num = fiddfj_num - 2*(math.pi)**2/((Nf+1)*L**2) \
                      * (k**2 * np.sin( i*k*math.pi/(Nf+1) )*np.sin(j*k*math.pi/(Nf+1) ) )
    return fiddfj_num


def  V(i,j):
    Vij_num = fn.dcroc(i,j)*(xvals[i]-L/2)**2/2
    return Vij_num

def  pot(x):
     return (xvals-L/2)**2/2
# asi entonces definimos las entradas de nuestra matriz

H = np.zeros((Nf,Nf))

for l in range(Nf):
    for m in range(Nf):
        H[l,m] = - hbar**2/(2*me) * fiddfj(l+1,m+1) + V(l,m)

# teniendo el Hamiltiniano, Diagonalizamos para obtener los autovalores

En, bn = np.linalg.eig(H)
En = np.round(En,2)

# De este resultado buscamos llegar a Psi(x,t) = sum cn phi_n(x) * exp(-i*En*t) en donde los cn se hallan como
# cn = sum(bn^i* * ai)


cn = []

for l in range(Nf):
    bn[:, l:l + 1] = bn[:, l:l + 1] / np.linalg.norm(bn[:, l:l + 1])
    cn.append(sum((bn[:,l:l+1].T)[0]*dn))

cn = np.array(cn)
cn = cn/ sum(cn**2)
# Formamos los autoestados en forma de función

phin = np.zeros((Nf,Nf))

for l in range(Nf):
    phinp = np.zeros((Nf,1))
    for m in range(Nf):
        phinp = phinp + bn[m,l]*np.array([f(xvals,m+1)]).T
    phin[:,l:l+1] = phinp/np.linalg.norm(phinp)

# Con esto ya podemos desarrollar la propagación temporal:


def psi(tt):
    Psi = np.zeros(Nf)
    for l in range(Nf):
        Psi = Psi + cn[l]*(phin[:,l:l+1].T)[0]*np.exp(-1j*En[l]*(tt-t0))
    return Psi

fig2,ax2 = plt.subplots()
ax2.set_xlim(0, L)
ax2.set_ylim(-0.3, 9)
ax2.plot(xvals,pot(xvals))

# ax2.plot(xvals,(phin[:,0:1].T)[0])
# for l in range(5):
#     ax2.plot(xvals,(phin[:,l:l+1].T)[0])

line_module, = ax2.plot([], [], lw=2, color = 'blue')
frame_text = ax2.text(0.02, 0.9, '', transform=ax2.transAxes, fontsize=12, color='red')

def init():
    line_module.set_data([],[])
    frame_text.set_text('')
    return line_module, frame_text

def animate(frames):
    tt = (frames+1) * 0.1
    y_vals = psi(tt)  # arguments: (x,t,p0,x0,t0)
    line_module.set_data(xvals, np.abs(y_vals))  # real part
    frame_text.set_text(f"{frames}")
    return line_module,frame_text


ani = animation.FuncAnimation(fig2, animate,init_func=init, frames=500, interval=70, blit=True)



plt.grid()
plt.show()

