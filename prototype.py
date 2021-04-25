import math
import numpy as np


def FFT_step1_reference(x, N, bursts, min_block_size):
    n = np.arange(min_block_size)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / min_block_size)
    x = np.transpose(x.reshape((min_block_size, -1, bursts)), (2, 0, 1))
    X = np.transpose(np.dot(M, x), (0, 2, 1))
    return X.flatten()


def FFT_step2_reference(X, N, bursts, min_block_size):
    X = X.reshape((min_block_size, -1, bursts))

    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2, :]
        X_odd = X[:, X.shape[1] // 2:, :]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[..., None, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.flatten()


def FFT_step1(x, N, bursts, min_block_size):
    """
    Explicitly organising the code the way it will be structured in CUDA
    """
    n = np.arange(min_block_size)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / min_block_size).flatten()

    y = np.empty(N * bursts, dtype=np.complex64)

    def subthread(burst):
        num_blocks = N // min_block_size
        for block in range(num_blocks):
            for k in range(min_block_size):
                y_k = 0
                for n in range(min_block_size):
                    elt = x[(n * num_blocks + block) * bursts + burst]
                    # twiddle = M[k * min_block_size + n]
                    exponent = -2 * np.pi * n * k / min_block_size
                    twiddle = math.cos(exponent) + 1j * math.sin(exponent)
                    y_k += elt * twiddle

                y[(k * num_blocks + block) * bursts + burst] = y_k

    for b in range(bursts):
        subthread(b)

    return y


def FFT_step2(X, N, bursts, min_block_size):
    """
    Explicitly organising the code the way it will be structured in CUDA
    """
    all_factors = np.exp(-1j * np.pi * np.arange(N // 2) / (N//2))

    def subthread(x, y, burst, num_blocks, block_size):
        half_block_size = block_size // 2
        for block in range(num_blocks):  # Row of X
            for i in range(half_block_size):  # Column of X
                even = x[(block * block_size + i) * bursts + burst]
                odd = x[(block * block_size + half_block_size + i) * bursts + burst]

                # odd *= all_factors[block * half_block_size]
                exponent = -2 * np.pi * block * half_block_size / N
                twiddle = math.cos(exponent) + 1j * math.sin(exponent)
                odd *= twiddle

                y[(block * half_block_size + i) * bursts + burst] = even + odd
                y[((num_blocks + block) * half_block_size + i) * bursts + burst] = even - odd

    num_blocks = min_block_size
    block_size = N // min_block_size

    while num_blocks < N:
        Y = np.empty_like(X)

        for b in range(bursts):
            subthread(X, Y, b, num_blocks, block_size)

        X = Y
        num_blocks *= 2
        block_size //= 2

    return X


def FFT_step1_transpose(x, N, bursts, min_block_size):
    n = np.arange(min_block_size)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / min_block_size).flatten()

    y = np.empty(N * bursts, dtype=np.complex64)

    def subthread(burst):
        num_blocks = N // min_block_size
        for block in range(num_blocks):
            for k in range(min_block_size):
                y_k = 0
                for n in range(min_block_size):
                    elt = x[(n * num_blocks + block) * bursts + burst]
                    # twiddle = M[k * min_block_size + n]
                    exponent = -2 * np.pi * n * k / min_block_size
                    twiddle = math.cos(exponent) + 1j * math.sin(exponent)
                    y_k += elt * twiddle

                # y[(k * num_blocks + block) * bursts + burst] = y_k
                y[burst * N + k + block * min_block_size] = y_k

    for b in range(bursts):
        subthread(b)

    return y


def FFT_step2_transpose(X, N, bursts, min_block_size):
    all_factors = np.exp(-1j * np.pi * np.arange(N // 2) / (N//2))

    # Preload into 'shared memory'
    num_blocks = N // min_block_size
    local = np.empty_like(X)
    for burst in range(bursts):
        for block in range(num_blocks):
            for element in range(min_block_size):
                val = X[burst * N + element + block * min_block_size]
                local[burst * N + block + element * num_blocks] = val


    num_blocks = min_block_size
    block_size = N // min_block_size
    half_block_size = block_size // 2

    while num_blocks < N:
        Y = np.empty_like(X)

        for burst in range(bursts):
            for block in range(num_blocks):
                for i in range(half_block_size):
                    even = local[burst * N + (block * block_size + i)]
                    odd = local[burst * N + (block * block_size + half_block_size + i)]
                    odd *= all_factors[block * half_block_size]
                    # exponent = -2 * np.pi * block * half_block_size / N
                    # twiddle = math.cos(exponent) + 1j * math.sin(exponent)
                    # odd *= twiddle

                    Y[burst * N + (block * half_block_size + i)] = even + odd
                    Y[burst * N + ((num_blocks + block) * half_block_size + i)] = even - odd

        local = Y
        num_blocks *= 2
        block_size = half_block_size
        half_block_size = block_size // 2

    return local


def FFT_packed_reference(X, N, bursts, min_block_size):
    X1 = X[..., ::2].astype(np.complex64)
    X2 = X[..., 1::2].astype(np.complex64)
    X = X1 + 1j * X2
    
    out = FFT_step1_reference(X, N // 2, bursts, min_block_size)
    out = FFT_step2_reference(out, N // 2, bursts, min_block_size)

    return out



if __name__ == '__main__':

    N = 512
    bursts = 50
    min_block_size = min(N, 32)
    epsilon = 10e-4

    X = np.asarray(np.random.random(N * bursts), dtype=float)

    out1_ref = FFT_step1_reference(X, N, bursts, min_block_size)
    out2_ref = FFT_step2_reference(out1_ref, N, bursts, min_block_size)
    out1_transpose_ref = np.transpose(out1_ref.reshape((min_block_size, -1, bursts)), (2, 1, 0)).flatten()
    out2_transpose_ref = np.transpose(out2_ref.reshape((N, bursts))).flatten()

    out1 = FFT_step1(X, N, bursts, min_block_size)
    out2 = FFT_step2(out1, N, bursts, min_block_size)
    out1_transpose = FFT_step1_transpose(X, N, bursts, min_block_size)
    out2_transpose = FFT_step2_transpose(out1_transpose_ref, N, bursts, min_block_size)

    out_packed = FFT_packed_reference(X, N, bursts, min_block_size)

    success = "\033[92mSuccess!\033[0m"
    failure = "\033[91mFailure :(\033[0m"

    print("Phase 1: ",
          success if np.allclose(out1, out1_ref, atol=epsilon) else failure)
    print("Phase 2: ",
          success if np.allclose(out2, out2_ref, atol=epsilon) else failure)
    print("Phase 1 transpose: ",
          success if np.allclose(out1_transpose, out1_transpose_ref, atol=epsilon) else failure)
    print("Phase 2 transpose: ",
          success if np.allclose(out2_transpose, out2_transpose_ref, atol=epsilon) else failure)
    print("Packed: ",
          success if np.allclose(out_packed, out2_ref, atol=epsilon) else failure)

