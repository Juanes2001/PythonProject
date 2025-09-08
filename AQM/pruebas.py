import numpy as np
import matplotlib.pyplot as ptl
import math
from scipy import integrate

res = 10
n=6
x = np.linspace(start=-1,stop=1,num=res)

dom = [-1,1]


#definicion de una función que me entrega una base de funciones polinomicas X^n
def base(var,n):
    fun = np.array([var**i for i in range(n)])
    return fun

fun = base(x,n)

print(fun)

def dot(dom,fun1,fun2):

    result1, err1 = integrate.quad(lambda var: fun1(var)*fun2(var), dom[0], dom[1])
    return result1

# print(fun[1])

# print(arr_fun[0](x))

# print(dot(dom,arr_fun[2],arr_fun[0]))

# a = np.array([1,2,3,4,5,6])
# print(np.array([]))
# print(np.concatenate((np.array([1]),a))  )
#
# print(np.transpose(a))
# print(fun(2))
# print(a*fun(2))
# results_err = [integrate.quad(lambda t: fun(t)[i], -1, 1) for i in range(len(fun(0)))]
# results = []
#
# for j in results_err:
#     results.append(j[0])
#
# print (results)

# fig, ax = ptl.subplots()
# ax.plot(x,fun(x)[0])
# ax.plot(x, fun(x)[1])
# ax.plot(x, fun(x)[2])
# ax.plot(x, fun(x)[3])
# ptl.grid()

fun_sum = lambda var: np.array([f(var) for f in arr_fun]).sum()

print(fun_sum(1))

