import numpy as np


def FFT_step1_reference(x, N, N_min=32):    
    # Perform an O[N^2] DFT on all sub-problems of size N-min
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    return X.flatten()

def FFT_step2_reference(X, N, N_min=32):
    X = X.reshape((N_min, N // N_min))

    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.flatten()


def FFT_step1(x, N, N_min=32):
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min).flatten()

    num_blocks = N // N_min
    X = np.empty(N, dtype=np.complex)

    for block in range(num_blocks):
        for k in range(N_min):
            X_k = 0
            for n in range(N_min):
                X_k += M[k * N_min + n] * x[n * num_blocks + block]
            X[k * num_blocks + block] = X_k

    return X


def FFT_step2(X, N, N_min=32):
    all_factors = np.exp(-1j * np.pi * np.arange(N // 2) / (N//2))
    num_blocks = N_min
    block_size = N // N_min
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



if __name__ == '__main__':

    N = 512
    N_min = min(N, 32)
    x = np.asarray(np.random.random(N), dtype=float)

    out1_reference = FFT_step1_reference(x, N, N_min)
    out2_reference = FFT_step2_reference(out1_reference, N, N_min)

    out1 = FFT_step1(x, N, N_min)
    out2 = FFT_step2(out1, N, N_min)

    success = "\033[92mSuccess!\033[0m"
    failure = "\033[91m Failure :(\033[0m"
    print("Phase1: ", success if np.allclose(out1, out1_reference) else failure)
    print("Phase2: ", success if np.allclose(out2, out2_reference) else failure)
