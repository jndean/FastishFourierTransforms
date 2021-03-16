import numpy as np


success = "\033[92mSuccess!\033[0m"
failure = "\033[91mFailure :(\033[0m"


def FFT_step1_reference(x, N, min_block_size):
    # Perform an O[N^2] DFT on all sub-problems of size N-min
    n = np.arange(min_block_size)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / min_block_size)
    X = np.dot(M, x.reshape((min_block_size, -1)))

    return X.flatten()


def FFT_step2_reference(X, N, min_block_size):
    X = X.reshape((min_block_size, N // min_block_size))

    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.flatten()


def FFT_step1(x, N, min_block_size):
    n = np.arange(min_block_size)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / min_block_size).flatten()

    num_blocks = N // min_block_size
    X = np.empty(N, dtype=complex)

    for block in range(num_blocks):
        for k in range(min_block_size):
            X_k = 0
            for n in range(min_block_size):
                X_k += M[k * min_block_size + n] * x[n * num_blocks + block]
            X[k * num_blocks + block] = X_k

    return X


def FFT_step2(X, N, min_block_size):
    all_factors = np.exp(-1j * np.pi * np.arange(N // 2) / (N//2))
    num_blocks = min_block_size
    block_size = N // min_block_size
    half_block_size = block_size // 2

    while num_blocks < N:
        Y = np.empty_like(X)

        for block in range(num_blocks):  # Row of X
            for i in range(half_block_size):  # Column of X
                even = X[block * block_size + i]
                odd = X[block * block_size + half_block_size + i]
                odd *= all_factors[block * half_block_size]
                Y[block * half_block_size + i] = even + odd
                Y[(num_blocks + block) * half_block_size + i] = even - odd
        
        X = Y
        num_blocks *= 2
        block_size //= 2
        half_block_size = block_size // 2

    return X


def batched_FFT_step1_reference(x, N, batches, min_block_size):
    """
    Batched processing, "vectorised" with numpy
    """
    n = np.arange(min_block_size)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / min_block_size)
    x = np.transpose(x.reshape((min_block_size, -1, batches)), (2, 0, 1))
    X = np.transpose(np.dot(M, x), (0, 2, 1))
    return X.flatten()


def batched_FFT_step2_reference(X, N, batches, min_block_size):
    """
    Batched processing, "vectorised" with numpy
    """
    X = X.reshape((min_block_size, -1, batches))

    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2, :]
        X_odd = X[:, X.shape[1] // 2:, :]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[..., None, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.flatten()


def batched_FFT_step1(x, N, batches, min_block_size):
    """
    Batched processing, explicitly organising the code the
    way it will be structured in CUDA
    """
    n = np.arange(min_block_size)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / min_block_size).flatten()

    y = np.empty(N * batches, dtype=complex)

    def subthread(batch):
        num_blocks = N // min_block_size
        for block in range(num_blocks):
            for k in range(min_block_size):
                y_k = 0
                for n in range(min_block_size):
                    elt = x[(n * num_blocks + block) * batches + batch]
                    twiddle = M[k * min_block_size + n]
                    y_k += elt * twiddle
                y[(k * num_blocks + block) * batches + batch] = y_k

    for b in range(batches):
        subthread(b)

    return y


def batched_FFT_step2(X, N, batches, min_block_size):
    """
    Batched processing, explicitly organising the code the
    way it will be structured in CUDA
    """
    all_factors = np.exp(-1j * np.pi * np.arange(N // 2) / (N//2))

    def subthread(x, y, batch, num_blocks, block_size):
        half_block_size = block_size // 2
        for block in range(num_blocks):  # Row of X
            for i in range(half_block_size):  # Column of X
                even = x[(block * block_size + i) * batches + batch]
                odd = x[(block * block_size + half_block_size + i) * batches + batch]
                odd *= all_factors[block * half_block_size]
                y[(block * half_block_size + i) * batches + batch] = even + odd
                y[((num_blocks + block) * half_block_size + i) * batches + batch] = even - odd

    num_blocks = min_block_size
    block_size = N // min_block_size

    while num_blocks < N:
        Y = np.empty_like(X)

        for b in range(batches):
            subthread(X, Y, b, num_blocks, block_size)

        X = Y
        num_blocks *= 2
        block_size //= 2

    return X


if __name__ == '__main__':

    N = 512
    batches = 50

    min_block_size = min(N, 32)
    X = np.asarray(np.random.random(N * batches), dtype=float)


    # Singular burst processing #
    x = X[::batches]

    out1_reference = FFT_step1_reference(x, N, min_block_size)
    out2_reference = FFT_step2_reference(out1_reference, N, min_block_size)

    out1 = FFT_step1(x, N, min_block_size)
    out2 = FFT_step2(out1, N, min_block_size)

    print("Phase 1: ", success if np.allclose(out1, out1_reference) else failure)
    print("Phase 2: ", success if np.allclose(out2, out2_reference) else failure)



    # Batched burst processing #
    batched_out1_reference = batched_FFT_step1_reference(X, N, batches, min_block_size)
    batched_out2_reference = batched_FFT_step2_reference(
        batched_out1_reference, N, batches, min_block_size)

    batched_out1 = batched_FFT_step1(X, N, batches, min_block_size)
    batched_out2 = batched_FFT_step2(batched_out1, N, batches, min_block_size)


    print("Batched Phase 1: ", 
        success if np.allclose(batched_out1, batched_out1_reference) else failure)
    print("Batched Phase 2: ",
        success if np.allclose(batched_out2, batched_out2_reference) else failure)


    # Final sanity check #
    X = X.reshape((N, batches))
    batched_out2 = batched_out2.reshape((N, batches))
    assert(np.allclose(batched_out2, np.fft.fft(X, axis=0)))
