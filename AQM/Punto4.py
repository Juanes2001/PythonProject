"""
    Aqui mostraremos el desarrollo del punto 4 , primero definiendo U como la matriz de rotación sobre la base
    ortonormal {|u1>,|u2>},  y luego Udag como la transpuerta de la matriz de rotación.

"""

import math
from sympy.core.random import random

#Importamos las librerias y funciones que podamos usar sobre la marcha
import functions as fn
import numpy as np

baseR2 = fn.baseRn(2)

u1 = np.array([baseR2[0]]).T
u2 = np.array([baseR2[1]]).T

#Definimos U como la matriz de rotacion, ya que asi esta definido el cambio de coordenadas

#Rotación sentido antihorario
def U(theta):

    rot_matrix =np.array([ [math.cos(theta*math.pi/180),-math.sin(theta*math.pi/180)],
                           [math.sin(theta*math.pi/180),math.cos(theta*math.pi/180)] ] )
    return rot_matrix

#Rotación sentido horario
def Udag(theta):

    rot_inv_matrix =np.array([ [math.cos(theta*math.pi/180),math.sin(theta*math.pi/180)],
                               [-math.sin(theta*math.pi/180),math.cos(theta*math.pi/180)] ] )
    return rot_inv_matrix

# Ahora para una matriz A definida como en el ejercicio usando cualquier numero random como representacion de las entradas de a

a = random()

A = np.array([[a,a],
              [a,-a]])

# Por lo que los autovalores de este serán, -(a2-lam2) =a2  --> lam2 =2a2 --> lam2 = +-sqrt(2)a
# Usamos la matriz de rotación U de tal forma que podamos hallar el theta para el cual se cumple que
# D = Udag*A*U,
# Se puede demostrar que esta transformacion es solo una doble rotación, en el cual [a,a] rota hasta [sqrt(2)a,0] y
# [a,-a] rota hasta [0,-sqrt(2)a], por lo que podemos hallar el angulo que se requiere para la tarea solo aplicando propidedades.
# de los vectores.

Au1 = A[:,0:1]
DA = np.array([[np.linalg.norm(Au1)],
               [0]])

#Usamos estos dos vectores para hallar el angulo entre ellos

theta = (180/math.pi)*math.acos(fn.dot(None,Au1,DA)/(np.linalg.norm(Au1)*np.linalg.norm(DA)))/2

print(f"Angulo de rotación = -{theta}")

# Veamos que en efecto si tenemos la diagonalizacion con este theta

D = np.round(Udag(theta)@A@U(theta)/a,2)

print(D)

# Sus autovalores normalizados entonces son :

print(f"lambda1 = {D[0][0]} y lambda2 = {D[1][1]}")

# Para hallar los auto estados del operador A los hallamos en las columnas de la matriz Udag en este caso, osea la matriz
# que se coloca a la izquierda de la matriz diagonal que acabamos de hallar.


print(Udag(-theta))

#Comprobamos que efectivamente si son autovectores.

A1 = A@U(theta)[:,0:1]/(math.sqrt(2)*a)
A2 = A@U(theta)[:,1:2]/(-math.sqrt(2)*a)

print(U(theta)[:,0:1])
print(A1)
print("//////////////////////////////////////////")
print(U(theta)[:,1:2])
print(A2)

# Por lo que solo basta describir los autovectores en la base original aplicando la rotacion inversa a los autovectores

u1auto = np.round(Udag(theta)@A1,2)
u2auto = np.round(Udag(theta)@A2,2)

print(u1auto)
print(u2auto)