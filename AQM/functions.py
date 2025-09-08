# aqui se colocaran las funciones correspondientes para facilitar la lectura de codigo en Gram_Schmidt.py

import numpy as np
import math
from scipy import integrate
import sympy as sp

# definicion del dominio a usar -1<= x <= 1

res = None
xdom = None

#definicion de una función que me entrega una base de funciones polinomicas X^n
def base(var,n):
    fun = np.array([var**i for i in range(n)])
    return fun

#definición del algoritmo de Gram_Schmidt, la entrada será una base de funciones
def Gram(dom, fun):
    # Queremos retornar ya la base pero ortogonalizada, por lo que el retorno sera otro vector de funciones

    # primero leemos cuantas funciones de la base son:
    len_base = len(fun)
    #aplicamos el algoritmo definiendo cada funcion por separado y luego uniendolas al final:
    arr_func = [fun[0]] # almacenamos la primera función de la nueva base ortogonal
    acum = np.zeros(len(dom))
    for k in range (1,len_base):
        for j in range(k):

            cross_point= dot(dom,arr_func[j],fun[k]) # producto punto cruzado
            norm_sqr = dot(dom,arr_func[j],arr_func[j]) # norma al cuadrado


            acum = acum + np.array((cross_point/norm_sqr)*arr_func[j])


        arr_func.append(fun[k] - acum)
        acum = np.zeros(len(dom))

    fun_ort = np.array(arr_func)
    base_ortN = norm(dom,fun_ort) ## Normalizamos la base resultante ya ortogonal

    return base_ortN



#definimos la función normalizar

def norm(dom,fun_ort):

    #contamos cual es la dimension de la base
    len_base = len(fun_ort)
    norm_sqr = []
    for i in range(len_base):

        norm_sqr.append(math.sqrt(dot(dom,fun_ort[i],fun_ort[i])))

    fun_ort_norm = np.array([fun_ort[i]/norm_sqr[i] for i in range(len_base)])
    return fun_ort_norm


#definimos la funcion producto punto:
def dot(dom,fun1,fun2):

    inte = integrate.simpson(fun1*fun2,dom)
    return inte









