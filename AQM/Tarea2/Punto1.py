#
import math

from sympy.core.random import random

#Importamos las librerias y funciones que podamos usar sobre la marcha
from AQM.Tarea2 import functions as fn
import numpy as np
import matplotlib.pyplot as plt

# Definimos una base bidimensional que haga las veces de los autoestados del Hamiltoniano. Base ortogonal.

dim = 2
base = fn.baseRn(dim)

u1 = np.array([base[0]]).T
u2 = np.array([base[1]]).T

# Definimos los estados no estacionarios:

mas   = 1/math.sqrt(2) * (u1+u2)
menos = 1/math.sqrt(2) * (u1-u2)

# Como tenemos estados de hamiltoniano, entonces tenemos estados de energia:

E1,E2 = -1,1

# Definimos la función de onda en t= 0 como el estado superpuesto |+> y el hamiltoniano

phi0 = mas
H = E1*u1@u1.T + E2*u2@u2.T

# Ahora para definir la funcion de onda en t>0 , esto se define con el operador funcional exp(iHt/hbar), pero este
# programarlo de forma exacta resultaria muy dificil, por lo que usamos algunos de los terminos de la serie exponencial
# Definimos la función propagador del hamiltoniano usando la serie exponencial truncada

n = 150
# Con esto ya podemos solo usar el operador en su forma matricial para hallar la propagacion en t>0


# Con el estado de a función de onda en t>0 vamos a calcular la probabilidad de que la función colapse en el estado
# combinado |+> y |->, C1 y C2 seran las constantes guardaran la informacion de probabilidad de cada estado |+> y |->


def prob1(order,H,phi0,t):
    global mas
    hamilt = np.zeros((2, 2))
    for i in range(order):
        hamilt = hamilt + (1 / math.factorial(i)) * (np.linalg.matrix_power(H, i)) * math.pow(t, i) * (1j) ** i

    phit = hamilt @ phi0
    prob =  abs(fn.dot(None, phit, mas))**2

    return prob

t = np.linspace(20,38,600)
prob1_arr = []

for i in range(len(t)):
    prob1_arr.append(prob1(n,H,phi0,t[i]))

figure,(ax1,ax2) = plt.subplots(1,2)

ax1.plot(t,prob1_arr)

def prob2(order,H,phi0,t):
    global menos
    hamilt = np.zeros((2, 2))
    for i in range(order):
        hamilt = hamilt + (1 / math.factorial(i)) * (np.linalg.matrix_power(H, i)) * math.pow(t, i) * (1j) ** i

    phit = hamilt @ phi0
    prob =  abs(fn.dot(None, phit, menos))**2

    return prob

prob2_arr = []

for i in range(len(t)):
    prob2_arr.append(prob2(n,H,phi0,t[i]))

ax1.plot(t,prob2_arr)
ax1.grid()


#Ahora vamos a comparar el resultado con el exacto, teniendo en cuenta que el operador propagador temporal usa todos los
#infinitos terminos de la serie

phit = 1/math.sqrt(2) * (  u1@np.exp([1j*E1*t]) + u2@np.exp([1j*E2*t]))

C1sq = []

for i in range(len(t)):
    C1sq.append(abs(fn.dot(None, phit[:,i:i+1], mas))**2)

ax2.plot(t,C1sq)

C2sq = []

for i in range(len(t)):
    C2sq.append(abs(fn.dot(None, phit[:,i:i+1], menos))**2)

ax2.plot(t,C2sq)
ax2.grid()
plt.show()









