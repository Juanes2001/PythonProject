# Aqui se encuentra todo el algoritmo de ortogonalizacion de Gram Schmidt en el cual
# se usan funciones establecidas en functions.py para facilitar el volumen de codigo y solo
# permitir en este archivo el calculo sencillo de ortogonalizacion de bases.
import matplotlib.pyplot as plt

# un primer paso sera traer todas las herramientas creadas para la ortogonalización

import functions as fn
import numpy as np
import matplotlib.pyplot as ptl
import pandas as pd
from tabulate import tabulate

num_base_functions = 5
fn.res = 10000
fn.xdom = np.linspace(-1,1,fn.res)

#Creamos la base de funciones polinomicas de Taylor centradas en cero.

base_taylor = fn.base(fn.xdom,num_base_functions) # como el ejecicio pide 5 funciones de base, entonces damos como entrada el numero 5 para que
                         # se generen 5 funciones de base partiendo desde la función 1.

#ploteamos las funciones:

fig, (ax1,ax2)= ptl.subplots(1,2)

len_base = len(base_taylor)
for i in range(len_base):
    ax1.plot(fn.xdom, base_taylor[i])
ax1.grid()


#Hallamos la ortogonormalización de la base:

base_ort = fn.Gram(fn.xdom,base_taylor)

#ploteamos la nueva base:

len_base = len(base_ort)
for i in range(len_base):
    ax2.plot(fn.xdom, base_ort[i])
ax2.grid()


# por ultimo colocamos en una tabla todos los productos cruzados y normas a modo de comparacion para mostrar que
# en efecto obtuvimos una base ortonormal.

# sacamos los productos puntos cruzados en una matriz.
cross_dots = np.zeros((num_base_functions,num_base_functions))

for i in range(num_base_functions):
    for j in range(num_base_functions):
        cross_dots[i][j] = round(fn.dot(fn.xdom,base_taylor[i],base_taylor[j]),2)

#TABLA DE PRODUCTOS CRUZADOS CON LA BASE DE TAYLOR

data_taylor = pd.DataFrame(cross_dots,columns=["1", "x", "x^2", "x^3", "x^4"],index=["1", "x", "x^2", "x^3", "x^4"])

data_taylor_norms = pd.DataFrame(cross_dots.diagonal(),index=["1", "x", "x^2", "x^3", "x^4"])


print(tabulate(data_taylor, headers="keys", tablefmt="fancy_grid"))
print(tabulate(data_taylor_norms, tablefmt="fancy_grid"))


# sacamos los productos puntos cruzados en una matriz.
cross_dots_norm = np.zeros((num_base_functions,num_base_functions))

for i in range(num_base_functions):
    for j in range(num_base_functions):
        cross_dots_norm[i][j] = abs(round(fn.dot(fn.xdom,base_ort[i],base_ort[j]),2))

#TABLA DE PRODUCTOS CRUZADOS CON LA BASE DE ortogonormal hallada

data_ortonorm = pd.DataFrame(cross_dots_norm,columns=["e0", "e1", "e2", "e3", "e4"],index=["e0", "e1", "e2", "e3", "e4"])

data_ortonorm_norms = pd.DataFrame(cross_dots_norm.diagonal(),index=["e0", "e1", "e2", "e3", "e4"])


print(tabulate(data_ortonorm, headers="keys", tablefmt="fancy_grid"))
print(tabulate(data_ortonorm_norms, tablefmt="fancy_grid"))

#Mostramos las graficas
plt.show()









