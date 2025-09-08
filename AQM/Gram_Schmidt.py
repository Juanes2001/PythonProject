# Aqui se encuentra todo el algoritmo de ortogonalizacion de Gram Schmidt en el cual
# se usan funciones establecidas en functions.py para facilitar el volumen de codigo y solo
# permitir en este archivo el calculo sencillo de ortogonalizacion de bases.
import matplotlib.pyplot as plt

# un primer paso sera traer todas las herramientas creadas para la ortogonalización

import functions as fn
import numpy as np
import matplotlib.pyplot as ptl
import sympy as sp

res = 1000
xdom = np.linspace(start=-1,stop=1,num=res)

#Creamos la base de funciones polinomicas de Taylor centradas en cero.

base_taylor = fn.base(5) # como el ejecicio pide 5 funciones de base, entonces damos como entrada el numero 5 para que
                         # se generen 5 funciones de base partiendo desde la función 1.

#ploteamos las funciones:

fig, (ax1,ax2)= ptl.subplots(1,2)

len_base = len(base_taylor(0))
for i in range(len_base):
    ax1.plot(xdom, base_taylor(xdom)[i])
ax1.grid()

#Hallamos la ortogonormalización de la base:

base_ort = fn.Gram([xdom[0],xdom[-1]],base_taylor)

#ploteamos la nueva base:

len_base = len(base_ort(0))
for i in range(len_base):
    ax2.plot(xdom, base_ort(xdom)[i])
ax2.grid()

plt.show()









