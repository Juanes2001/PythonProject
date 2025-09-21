#
import math

from sympy.core.random import random

#Importamos las librerias y funciones que podamos usar sobre la marcha
from AQM.Tarea2 import functions as fn
import numpy as np

# Definimos una base bidimensional que haga las veces de los autoestados del Hamiltoniano. Base ortogonal.

dim = 2
base = fn.baseRn(dim)

u1 = np.array([base[0]]).T
u2 = np.array([base[1]]).T

# Definimos los estados no estacionarios:

mas   = 1/math.sqrt(2) * (u1+u2)
menos = 1/math.sqrt(2) * (u1-u2)

# Como tenemos estados de hamiltoniano, entonces tenemos estados de energia:

E1,E2 = random(),random()

# Definimos la función de onda en t= 0 como el estado superpuesto |+> y el hamiltoniano

phi0 = mas
H = E1*u1@u1.T + E2*u2@u2.T

# Ahora para definir la funcion de onda en t>0 , esto se define con el operador funcional exp(iHt/hbar), pero este
# programarlo de forma exacta resultaria muy dificil, por lo que usamos algunos de los terminos de la serie exponencial
# Definimos la función propagador del hamiltoniano usando la serie exponencial truncada

n = 10
# Con esto ya podemos solo usar el operador en su forma matricial para hallar la propagacion en t>0

def prob1(order,H,phi0,t):
    global mas
    c = np.zeros((2, 2))
    for i in range(order):
        c = c + (1 / math.factorial(i)) * (np.linalg.matrix_power(H, i)) * math.pow(t, i) * (1j) ** i

    phit = c @ phi0
    prob =  fn.dot(None, phit, mas )

    return c @ phi0

# Con el estado de a función de onda en t>0 vamos a calcular la probabilidad de que la función colapse en el estado
# combinado |+> y |->, C1 y C2 seran las constantes guardaran la informacion de probabilidad de cada estado |+> y |->

def C1(order,t):
    return fn.dot(None, phit, mas )










