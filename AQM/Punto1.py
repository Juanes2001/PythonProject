#Para trabajar usando Python el punto 1, como la base bidimensional no esta definida,
# se tratara la base como si fueran los vectores  de R2, osea e1 y e2, por lo que el Ket
# estara asociado  con el vector columna y el bra estara asociado al vector fila.
from mpmath import eig
from sympy.core.random import random

#Importamos las librerias y funciones que podamos usar sobre la marcha
import functions as fn
import numpy as np
import matplotlib.pyplot as ptl
import pandas as pd
from tabulate import tabulate
import sympy as sp

dim = 2
base = fn.baseRn(dim) # Con esta funcion obtenemos una base del plano que va a representar a nuestra base
                      #  de funciones.

# Teniendo la base podemos definir u1 y u2

u1 = np.array([base[0]]).T
u2 = np.array([base[1]]).T

# Definimos el operador A y B

A = -u1 @ u2.T -u2 @ u1.T # Teniendo en cuenta que la representacion de los KET son columna y los BRA son vectores
                          # fila.

B = 3*u1 @ u1.T +3*u2 @ u2.T -u1 @ u2.T -u2 @ u1.T

## Veremos si los operadores conmutan.

Phi = random()*u1 -random()*u2 #Generamos una combinacion lineal random
conmut =(A@B)-(B@A)

print(f"[A,B]*phi = {(conmut@Phi).sum()}")

# Buscamos los autoestados del operador B diagonalizando la matriz de B

eigvals1, eigvec1 = np.linalg.eig(B)

#Veamos que en efecto se cumple que hayamos los autovectores con sus autovalores

b1 = eigvals1[0]
b2 = eigvals1[1]

phi1 = (eigvec1.T)[0].T
phi2 = (eigvec1.T)[1].T


print (phi1,"\n",(B@phi1)/b1)
print("#####################")
print (phi2,"\n",(B@phi2)/b2)

#Si A*phi1 y A*phi2 son tambien autovectores de B significa que phi1 y phi2 son autoestados de A
#Por lo que hacemos el mismo procedimiento para A

eigvals2, eigvec2 = np.linalg.eig(A)

#Veamos que en efecto se cumple que hayamos los autovectores con sus autovalores

a1 = eigvals2[0]
a2 = eigvals2[1]

#Sin usar los autovectores que me entregan para A, usando solo los de B, veamos si nos da iguales a los
# autovectores de B

print (phi1,"\n",(A@phi1)/a1)
print("#####################")
print (phi2,"\n",(A@phi2)/a2)

#Vemos que en efecto obtenemos los mismos autovectores de B usando los obtenidos para B suponiendo que son
# los mismos autovectores en A, esto es debido a que tanto el operador A como B son matrices hermiticas y ademas
#conmutan









