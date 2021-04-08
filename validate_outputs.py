import ctypes
import numpy as np
import subprocess
import sys

import prototype


N = 512
bursts = 10
atol = 0.5


def equal(complex_array, float_array):
    if (np.allclose(np.real(complex_array), float_array[0::2], atol=atol) and
            np.allclose(np.imag(complex_array), float_array[1::2], atol=atol)):
        return "\033[92mSuccess!\033[0m"
    return "\033[91mFailure :(\033[0m"


if __name__ == '__main__':

    if subprocess.run(["make", "FiFT.so"]).returncode != 0:
        sys.exit("Failed to compile FiFT")

    fift = ctypes.cdll.LoadLibrary("./FiFT.so")
    min_block_size = ctypes.c_int.in_dll(fift, "base_block").value

    data = np.frombuffer(np.random.bytes(N * bursts), dtype=np.uint8)

    reference1 = prototype.FFT_step1(data, N, bursts, min_block_size)
    reference2 = prototype.FFT_step2(reference1, N, bursts, min_block_size)
    ref1_transpose = prototype.FFT_step1_transpose(data, N, bursts, min_block_size)
    ref2_transpose = prototype.FFT_step2_transpose(ref1_transpose, N, bursts, min_block_size)

    out1 = np.empty(N * bursts * 2, dtype=np.float32)
    out2 = np.empty(N * bursts * 2, dtype=np.float32)
    out1_transpose = np.empty(N * bursts * 2, dtype=np.float32)
    out2_transpose = np.empty(N * bursts * 2, dtype=np.float32)

    fift.test_step1(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N, bursts
    )
    fift.test_run(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N, bursts
    )
    fift.test_step1_transpose(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out1_transpose.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N, bursts
    )
    fift.test_step2_transpose(
        out1_transpose.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out2_transpose.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N, bursts
    )

    print('Step 1:', equal(reference1, out1))
    print('Step 1 & 2:', equal(reference2, out2))
    print('Step 1 transpose:', equal(ref1_transpose, out1_transpose))
    print('Step 2 transpose:', equal(ref2_transpose, out2_transpose))
