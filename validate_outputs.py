import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import subprocess
import sys

import prototype


if __name__ == '__main__':
    
    result = subprocess.run(["make", "FiFT.so"])
    if result.returncode != 0:
        sys.exit("Failed to compile FiFT")

    fift_lib = ctypes.cdll.LoadLibrary("./FiFT.so")
    fift = fift_lib.test
    fift.restype = None
    """fift.argtypes = [
        ndpointer(ctypes.c_ubyte, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t
    ]"""

    N = 512
    batches = 50
    min_block_size = 16

    idata = np.frombuffer(np.random.bytes(N * batches), dtype=np.uint8)
    odata = np.empty(N * batches * 2, dtype=np.float32)

    fift(idata.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
         odata.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
         N,
         batches)

    reference = prototype.batched_FFT_step1_reference(idata, N, batches, min_block_size)
    #print(np.allclose(reference, odata))

    print(reference[0], odata[0])
