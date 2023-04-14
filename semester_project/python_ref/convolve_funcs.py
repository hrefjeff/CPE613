""" =================================================================
Author      : Jeffrey Allen
Class       : CS613, General Purpose GPU Programming
Assignment  : Semester Project
Description : Implementations of different convolution methods

Table of Contents:

1. Time Domain
2. FFT
3. Overlap Add
4. Overlap Save
================================================================="""

import numpy as np
from numpy import exp
from scipy import signal

"""=== 1. Time Domain ==="""

def convolve_time_domain(arr1, arr2):
    """ Naive convolve method that convoles arr2 onto arr1
        The Scientist and Engineer's Guide to Digital Signal Processing
        provides an explanation of this on Pg. 116 "The Output Side Algorithm"
    """

    # Sets up arrays to work with
    arr1_len = len(arr1)
    arr2_len = len(arr2)
    output_len = arr1_len + arr2_len - 1
    output = [0.0] * (output_len)

    for idxInput in range(output_len): # Loop over input array
        output[idxInput] = 0.0
        for idxFilter in range(arr2_len): # Loop over convolution filter
            position = idxInput - idxFilter
            # Check if the positions to add are within the bounds
            if (position < 0) or (position >= arr1_len):
                pass # Outside of the bounds of calculation, do nothing
            else:
                output[idxInput] = output[idxInput] + arr2[idxFilter] * arr1[idxInput - idxFilter]
        
    return output

def convolve_time_domain_np(arr1, arr2):
    """ Convolve 2 arrays with np's convolve 
        https://numpy.org/doc/stable/reference/generated/numpy.convolve.html#numpy-convolve
    """
    return np.convolve(arr1, arr2).tolist()

"""=== 2. FFT ==="""

def W_N(N,k,n):
    return exp(-2j * np.pi * k * n / N)

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x
    http://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = W_N(N, k, n)
    return np.dot(M, x)

def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT
    https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/FFT_Demo.html
    """
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)
    
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n_idx = np.arange(N_min)
    k_idx = n_idx[:, None]
    D = W_N(N_min, n_idx, k_idx)
    X = np.dot(D, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = W_N(X.shape[0], np.arange(X.shape[0]), 0.5)[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()

def pad_zeros_to(x, new_length):
    """Append new_length - x.shape[0] zeros to x's end via copy."""
    output = np.zeros((new_length,))
    output[:x.shape[0]] = x
    return output

def next_power_of_2(n):
    return 1 << (int(np.log2(n - 1)) + 1)

def convolve_fft(x_list, h_list, K=None):
    """
    https://thewolfsound.com/fast-convolution-fft-based-overlap-add-overlap-save-partitioned/
    """
    x = np.array(x_list)
    h = np.array(h_list)
    Nx = x.shape[0] # Simply gets the number of elements in the array
    Nh = h.shape[0]
    Ny = Nx + Nh - 1 # output length

    # Make K smallest optimal
    if K is None:
        K = next_power_of_2(Ny)

    # Calculate the fast Fourier transforms 
    # of the time-domain signals
    X = FFT_vectorized(pad_zeros_to(x, K))
    H = FFT_vectorized(pad_zeros_to(h, K))

    print("====Printing signal====")
    for i in range(0,5):
        print(X[i])

    print("====Filter signal====")
    for i in range(0,5):
        print(H[i])

    # Perform circular convolution in the frequency domain
    Y = np.multiply(X, H)

    # Go back to time domain
    y = np.real(np.fft.ifft(Y))

    # Trim the signal to the expected length
    return y[:Ny]

def convolve_fft_sp(arr1, arr2):
    """
    This is generally much faster than convolve for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).
    """
    return signal.fftconvolve(arr1, arr2, mode='full')

"""=== 3. Overlap Add (TODO) ==="""

def convolve_oa_sp(arr1, arr2):
    """ Convolve using overlap add method
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.oaconvolve.html#scipy.signal.oaconvolve
    This is generally much faster than convolve for large arrays (n > ~500),
    and generally much faster than fftconvolve when one array is much larger
    than the other, but can be slower when only a few output values are needed
    or when the arrays are very similar in shape, and can only output float
    arrays (int or object array inputs will be cast to float).
    """
    return signal.oaconvolve(arr1, arr2)

"""=== 4. Overlap Save (TODO) ==="""