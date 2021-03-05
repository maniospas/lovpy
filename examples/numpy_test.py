from logipy.wrappers import LogipyPrimitive, logipy_call
import numpy as np

x = logipy_call(np.array,[.1, .2, .9])
x += 1
logipy_call(print,logipy_call(repr,x[1]))
