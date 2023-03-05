import numpy as np

def convolve_time_domain(arr1, arr2, radius):
    """ Convolve two arrays using the method from the book
        Page 153, section 7.1
    """

    # Create a new array to hold the result
    arr1_len = len(arr1)
    arr2_len = len(arr2)
    result = [0] * (arr1_len)

    for idxInput in range(arr1_len): # Loop over input array
        sum = 0
        for idxFilter in range(arr2_len): # Loop over convolution filter
            # Check if the filter element aligns with an input
            # element within the bounds of the input array
            inputPos = idxInput + (idxFilter - radius)
            if inputPos < 0 or inputPos >= arr1_len:
                sum += 0
            else:
                sum += arr1[inputPos] * arr2[idxFilter]
        result[idxInput] = sum

    return result

def fft_v(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0] # Get the number of elements in the array
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")

    N_min = min(N, 2)
    
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min) # Calculate the DFT matrix
    X = np.dot(M, x.reshape((N_min, -1))) # Calculate the DFT
    while X.shape[0] < N:
            X_even = X[:, :int(X.shape[1] / 2)]
            X_odd = X[:, int(X.shape[1] / 2):]
            terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                            / X.shape[0])[:, None]
            X = np.vstack([X_even + terms * X_odd,
                        X_even - terms * X_odd])
    return X.ravel()

def convolve_fast_fourier_transform(arr1, arr2):
    """ Convolve two arrays using the fast fourier transform
        The convolution in the time domain is equivalent to
        multiplication in the frequency domain
    """

    # Create a new array to hold the result
    result = [0] * len(arr1)

    # Calculate the FFT of the input arrays
    fft1 = fft_v(arr1)
    fft2 = fft_v(arr2)

    # Multiply the FFTs
    fft_result = fft1 * fft2

    # Calculate the inverse FFT
    result = np.real(fft_v(fft_result))
    
    return result
