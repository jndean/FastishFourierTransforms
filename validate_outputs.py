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

    N = 512
    batches = 50
    min_block_size = 16

    idata = np.frombuffer(np.random.bytes(N * batches), dtype=np.uint8)
    odata = np.empty(N * batches * 2, dtype=np.float32)

    fift(idata.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
         odata.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
         N,
         batches)

    success = "\033[92mSuccess!\033[0m"
    failure = "\033[91mFailure :(\033[0m"

    reference = prototype.FFT_step1(idata, N, batches, min_block_size)
    print('Step 1:', success if (
        np.allclose(np.real(reference), odata[0::2], atol=0.01) and
        np.allclose(np.imag(reference), odata[1::2], atol=0.01)
    ) else failure)

