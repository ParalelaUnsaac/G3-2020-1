

import os
os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/local/cuda-10.0/nvvm/libdevice"
os.environ['NUMBAPRO_NVVM'] = "/usr/local/cuda-10.0/nvvm/lib64/libnvvm.so"

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

@vectorize(['float32(float32)'], target='cuda')
def gpu_sqrt(x):
    return math.sqrt(x)
cpu_sqrt(x)
