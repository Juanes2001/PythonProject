# aqui se colocaran las funciones correspondientes para facilitar la lectura de codigo en Gram_Schmidt.py

import numpy as np
import matplotlib.pyplot as ptl
import math
from scipy import integrate

# definicion del dominio a usar -1<= x <= 1

res = 1000
xdom = np.linspace(start=-1,stop=1,num=res)


#definicion de una función que me entrega una base de funciones polinomicas X^n
def base(n):
    fun =lambda var: np.array([var**i for i in range(n)])
    return fun

#definición del algoritmo de Gram_Schmidt, la entrada será una base de funciones
def Gram(dom, fun):
    # Queremos retornar ya la base pero ortogonalizada, por lo que el retorno sera otro vector de funciones

    # primero leemos cuantas funciones de la base son:
    len_base = len(fun(0))
    fun_vec = [lambda var, i=i: fun(var)[i] for i in range(len_base)]
    #aplicamos el algoritmo definiendo cada funcion por separado y luego uniendolas al final:
    arr_func = [lambda var: fun(var)[0]] # almacenamos la primera función
    acum =np.array([])
    for k in range (1,len_base):
        for j in range(k):

            cross_point= dot(dom,arr_func[j],fun_vec[k]) # producto punto cruzado
            norm_sqr = dot(dom,arr_func[j],arr_func[j]) # norma al cuadrado

            acum.concatenate((acum,np.array([lambda var: (cross_point/norm_sqr)*arr_func[j](var)])))


        arr_func.append(lambda var: fun(var)[k] - sum(acum[:](var)))

    fun_ort = lambda var: np.array([arr_func[i](var) for i in range(len_base)])

    base_ortN = norm(dom,fun_ort) ## Normalizamos la base resultante ya ortogonal

    return base_ortN



#definimos la función normalizar

def norm(dom,fun_ort):

    #contamos cual es la dimension de la base
    len_base = len(fun_ort(0))
    arr_funct = [lambda var, i=i: fun_ort(var)[i] for i in range(len_base)]
    norm_sqr = []
    for i in range(len_base):

        result1 = dot(dom,arr_funct[i],arr_funct[i])
        norm_sqr.append(math.sqrt(result1))

    fun_ort_norm = lambda var: np.array([fun_ort(var)[i]/norm_sqr[i] for i in range(len_base)])
    return fun_ort_norm


#definimos la funcion producto punto:

def dot(dom,fun1,fun2):

    result1, err1 = integrate.quad(lambda var: fun1(var)*fun2(var), dom[0], dom[1])
    return result1

#definimos la función para plotear en una grafica o conjunto de graficas
def plot(dom, fun):

    len_base = len(fun(0))
    fig, ax = ptl.subplots()
    for i in range(len_base):
        ax.plot(dom, fun(dom)[i])

    ptl.grid()
    ptl.show()







