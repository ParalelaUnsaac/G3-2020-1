import math
import timeit
import os
import numpy as np
os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/local/cuda-10.0/nvvm/libdevice"
os.environ['NUMBAPRO_NVVM'] = "/usr/local/cuda-10.0/nvvm/lib64/libnvvm.so"


x = np.arange(10)
x**2

np.exp(x)

import math
x = np.arange(int(1e6))
timeit.timeit(np.sqrt(x))
timeit.timeit([math.sqrt(xx) for xx in x])

import numpy as np
npoints = int(1e7)
a = np.arange(npoints, dtype=np.float32)

import math
from numba import vectorize

@vectorize
def cpu_sqrt(x):
  return math.sqrt(x)

cpu_sqrt(x)

import math
from numba import vectorize


