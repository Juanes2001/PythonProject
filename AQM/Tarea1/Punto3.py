"""
Consideramos el espacio desarollado por una base ortonormal {|n>}n = 0,1,2,3,4 , estan los operadores generacion y
destrucción, por lo que usando vectores base en Rn como ejemplo, construiremos la matriz de estos operadores

Luego definiremos la matriz N como la multiplicacion de adag*a. SImplemente tenemos que
aplicar para cada operador <i|a|j>,<i|adag|j>,<i|N|j>

"""
import math

#Importamos las librerias y funciones que podamos usar sobre la marcha
from AQM.Tarea1 import functions as fn
import numpy as np

#~Definimos una base de R5 ortogonal

baseR5 = fn.baseRn(5)

print(baseR5)

u0 = np.array([baseR5[0]]).T
u1 = np.array([baseR5[1]]).T
u2 = np.array([baseR5[2]]).T
u3 = np.array([baseR5[3]]).T
u4 = np.array([baseR5[4]]).T



# definimos el operador a desde su definicion en el ejecicio.

a = u0@u1.T + math.sqrt(2)*u1@u2.T + math.sqrt(3)*u2@u3.T + math.sqrt(4)*u3@u4.T

print(a)

#Similar para adag

adag = u1@u0.T + math.sqrt(2)*u2@u1.T + math.sqrt(3)*u3@u2.T + math.sqrt(4)*u4@u3.T

print(adag)


#Por ultimo definimos N como la multiplicacion de adag*a

N = adag@a

print(N)
