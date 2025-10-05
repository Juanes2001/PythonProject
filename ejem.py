import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL.ImageColor import colormap
from matplotlib import colors, colors
from matplotlib.pyplot import colormaps
from sympy.codegen.ast import integer


z=np.ones((3,3))
print(z)

z[:,0:1] = np.array([[1,2,3]]).T*np.array([[1,2,3]]).T
print(z)
print((z[:,0:1]))