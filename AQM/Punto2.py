#Para trabajar usando Python el punto 2, como la base tridimensional no esta definida,
# se tratará la base como si fueran los vectores  de R3, osea e1 , e2 y e3, por lo que el Ket
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


#Similar al punto 1  , definimos una base tridimensional

dim = 3
base = fn.baseRn(dim)

#Mostramos la representacion matricial de cada KET y cada BRA

u1 = np.array([base[0]]).T
u2 = np.array([base[1]]).T
u3 = np.array([base[2]]).T

print("|u1>=")
print (u1)

print("|u2>=")
print (u2)

print("|u3>=")
print (u3)

#Y ahora la representacion de los BRAs

print(f"<u1|={u1.T}")
print(f"<u2|={u2.T}")
print(f"<u3|={u3.T}")


#Similar a como se hizo en el ejercicio del punto 1, calculamos la matriz A y B en la base ui

A = u1@u1.T + u2@u2.T + u1@u2.T + u2@u1.T + u2@u3.T + u3@u2.T - u1@u3.T - u3@u1.T
B = 3*u1@u1.T + 3*u2@u2.T + 2*u3@u3.T + u1@u2.T + u2@u1.T

print(A)
print(B)

#Veamos de nuevo si las matrices conmutan para un vector cualquiera Phi

Phi = random()*u1 -random()*u2 +random()*u3
conmut = A@B - B@A

print(f"[A,B]*phi = {(conmut@Phi).sum()}")

#Encontremos los autovalores y autovectores de A y B

eigvalsA, eigvecA = np.linalg.eig(A)

eigvalsA = np.round(eigvalsA,3)
eigvecA  = np.round(eigvecA,3)

#Veamos que en efecto se cumple que hayamos los autovectores con sus autovalores

a1 = eigvalsA[0]
a2 = eigvalsA[1]
a3 = eigvalsA[2]

phiA1 = np.array([(eigvecA.T)[0]]).T
phiA2 = np.array([(eigvecA.T)[1]]).T
phiA3 = np.array([(eigvecA.T)[2]]).T

print(a1,a2,a3)

print(phiA1)
print(phiA2)
print(phiA3)

eigvalsB, eigvecB =  np.linalg.eig(B)

eigvalsB = np.round(eigvalsB,3)
eigvecB  = np.round(eigvecB,3)

## Hacemos lo mismo pero para B

b1 = eigvalsB[0]
b2 = eigvalsB[1]
b3 = eigvalsB[2]

phiB1 = np.array([(eigvecB.T)[0]]).T
phiB2 = np.array([(eigvecB.T)[1]]).T
phiB3 = np.array([(eigvecB.T)[2]]).T

print(b1,b2,b3)

print(phiB1)
print(phiB2)
print(phiB3)


##Lo que sigue es entonces representar la matriz de A usando la base de autoestados de B, y viceversa .
#Para ello Aplicamos A' = SAS, en donde S se contruye como <ui|phiBj> en donde u es la base vieja y phiBj
# la base nueva de autoestados de la diagonalizacion de B

base_old = [u1,u2,u3]
base_newfor_A = [phiB1,phiB2,phiB3]

Su_B = fn.Btransform_old_new(base_old,base_newfor_A)

Aprime = Su_B.T@A@Su_B

#Hacemos lo mismo para B pero usando la base de autoestados de A

base_newfor_B = [phiA1,phiA2,phiA3]

Su_A = fn.Btransform_old_new(base_old,base_newfor_B)

Bprime = Su_A.T@B@Su_A


print (f"A'= \n{np.round(Aprime,2)}")
print (f"B'= \n{np.round(Bprime,2)}")


# Como el sub espacio de los vectores del autovalor degenerado no es ortogonal, se busca llegar a una combinacion lineal
# de estos vectores para que puedan ser autoestados tanto de B como de A debido al teorema 4. Tomamos solo los autoestados de B
# con autovalor degenerado y aplicamos el metodo de Gram-Schmidt

