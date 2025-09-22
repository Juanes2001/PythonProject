# En este ejercicio a diferencia del anterior, los estados |1> y |2> ya no son autoestados del hamiltoniano gracias a la
# aparición de un potencial estacionario

import math

from sympy.core.random import random

#Importamos las librerias y funciones que podamos usar sobre la marcha
from AQM.Tarea2 import functions as fn
import numpy as np
import matplotlib.pyplot as plt

# Definimos una base bidimensional que haga las veces de los ahora estados ligados |1> y |2> del Hamiltoniano. Base ortogonal.

dim = 2
base = fn.baseRn(dim)

u1 = np.array([base[0]]).T
u2 = np.array([base[1]]).T

# con esto definimos el Hamiltoniano, usando los valores del ejercicio como:
E1,E2,W = -1,1,2

H = E1*u1@u1.T + E2*u2@u2.T + W*(u1@u2.T+u2@u1.T)

# El sistema se haya en el estado |1> en t = 0 , por lo que definimos phi0 como este estado

phi0 = u1

# ahora utilizamos el propagador temporal para analizar la probabilidad en cada estado subsecuente, el problema es que como
# |1> y |2> no son autoestados no H , hay que diagonalizar H y encontrar los nuevos autoestados, Luego escribir |1> y |2>
# desarrollados en la nueva base de autoestados de H.

eigvalsH, eigvectsH = np.linalg.eig(H)

E1prime,E2prime = eigvalsH[0],eigvalsH[1]

u1auto,u2auto = eigvectsH[:,0:1], eigvectsH[:,1:2]

# Hallados los autoestados de H y sus nuevos estados de energia, entonces solo escribimos u1 en terminos de u1auto y
# u2auto, ya que la condicion inicial se haya en el estado |1>

"""
Podemos escribir el sistema como  

|1> = C1*|u1auto>+C2*|u2auto>
"""
# Resolvemos el sistema

autoMatrx = np.column_stack((u1auto,u2auto))

consts = autoMatrx.T@u1

#Comprobamos que la combinacion esta bien hecha y reconstruye correctamente el vector u1

print(consts[0][0]*u1auto+ consts[1][0]*u2auto )
print(consts)

# Ya podemos entonces representar phi0 en terminos de una compinacion de los autoestados del hamiltoniano

phi0 = consts[0][0]*u1auto+ consts[1][0]*u2auto


#Ahora si aplicamos la propagación
def prob1(order,H,phi0,t):
    global u1
    hamilt = np.zeros((2, 2))
    for i in range(order):
        hamilt = hamilt + (1 / math.factorial(i)) * (np.linalg.matrix_power(H, i)) * math.pow(t, i) * (-1j) ** i

    phit = hamilt @ phi0
    prob =  abs(fn.dot(None, phit, u1))**2

    return prob


n = 210
t = np.linspace(0,9.6,600)
prob1_arr = []

for i in range(len(t)):
    prob1_arr.append(prob1(n,H,phi0,t[i]))

figure,(ax1,ax2) = plt.subplots(1,2)

ax1.plot(t,prob1_arr)
ax1.grid()


def prob2(order,H,phi0,t):
    global u1
    hamilt = np.zeros((2, 2))
    for i in range(order):
        hamilt = hamilt + (1 / math.factorial(i)) * (np.linalg.matrix_power(H, i)) * math.pow(t, i) * (-1j) ** i

    phit = hamilt @ phi0
    prob =  abs(fn.dot(None, phit, u2))**2

    return prob

prob2_arr = []

for i in range(len(t)):
    prob2_arr.append(prob2(n,H,phi0,t[i]))

ax1.plot(t,prob2_arr)

# Ahora calculamos las soluciones exactas usando el desarrollo completo en series


phit =  (consts[0][0]*u1auto@np.exp([-1j*E1prime*t])+ consts[1][0]*u2auto@np.exp([-1j*E2prime*t]))

C1sq = []

for i in range(len(t)):
    C1sq.append(abs(fn.dot(None, phit[:,i:i+1], u1))**2)

ax2.plot(t,C1sq)

C2sq = []

for i in range(len(t)):
    C2sq.append(abs(fn.dot(None, phit[:,i:i+1], u2))**2)

ax2.plot(t,C2sq)
ax2.grid()
plt.show()