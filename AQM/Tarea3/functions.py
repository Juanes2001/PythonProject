# aqui se colocaran las funciones correspondientes para facilitar la lectura de codigo en Gram_Schmidt.py

import numpy as np
import math
from scipy import integrate
import sympy as sp
from scipy import constants as const

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

    if dom != None :
        # Si estamos aqui es porque queremos ortogonalizar unos vectores en forma de funciones.
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
    else:
        # Si estamos aqui es porque queremos ortogonalizar un conjunto de vectores de Rn y no un espacio de funciones.

        # fun contendra todos los vectores a ortogonalizar
        len_base = len(fun)
        # aplicamos el algoritmo definiendo cada funcion por separado y luego uniendolas al final:
        arr_func = [fun[0]]  # almacenamos la primera función de la nueva base ortogonal
        acum = np.zeros(len(fun[0]))
        for k in range(1, len_base):
            for j in range(k):
                cross_point = dot(None, arr_func[j], fun[k])  # producto punto cruzado
                norm_sqr = dot(None, arr_func[j], arr_func[j])  # norma al cuadrado

                acum = acum + np.array((cross_point / norm_sqr) * arr_func[j])

            arr_func.append(fun[k] - acum)
            acum = np.zeros(len(fun[0]))

        fun_ort = np.array(arr_func)
        base_ortN = norm(dom, fun_ort)  ## Normalizamos la base resultante ya ortogonal

        return base_ortN


#definimos la función normalizar

def norm(dom,fun_ort):

    #contamos cual es la dimension de la base
    if dom != None:
        len_base = len(fun_ort)
        norm_sqr = []
        for i in range(len_base):

            norm_sqr.append(math.sqrt(dot(dom,fun_ort[i],fun_ort[i])))

        fun_ort_norm = np.array([fun_ort[i]/norm_sqr[i] for i in range(len_base)])
        return fun_ort_norm
    else:
        #Si estamos aqui es porque queremos normalizar una base ortogonalizada de Rn
        len_base = len(fun_ort)
        norm_sqr = []
        for i in range(len_base):
            norm_sqr.append(math.sqrt(dot(None, fun_ort[i], fun_ort[i])))

        fun_ort_norm = np.array([fun_ort[i] / norm_sqr[i] for i in range(len_base)])
        return fun_ort_norm


#definimos la funcion producto punto:
def dot(dom,fun1,fun2):
    if dom.any() != None:
        inte = integrate.simpson(fun1*fun2,dom)
        return inte
    else:
        # SI estamos aqui es porque fun1 y fun2 se tratan de vectores
        dot_prod = np.sum(fun1*fun2)
        return dot_prod

#////////////////////////////////////////////////////////////////////////////////////////////


#Definimos la funcion delta de croniquer
def dcroc(i, j):
    if i == j:
        return 1
    else:
        return 0


def baseRn(dim):
    base = []
    for i in range(1, dim + 1):
        base.append([dcroc(j, i) for j in range(1, dim + 1)])
    return np.array(base)

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def Btransform_old_new(old,new):
    dim = len(old)
    S = []
    for i in range(dim):
        S.append([(old[i].T @ new[j]).sum() for j in range(dim)])
    S =  np.array(S)
    return S


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def cos(theta):
    return math.cos(theta)

def sin(theta):
    return math.sin(theta)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



