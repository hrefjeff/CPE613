import numpy as np
from scipy import signal

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

def convolve_fft_sp(arr1, arr2):
    '''
    This is generally much faster than convolve for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).
    '''
    return signal.fftconvolve(arr1, arr2, mode='full')

def convolve_oa_sp(arr1, arr2):
    ''' Convolve using overlap add method
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.oaconvolve.html#scipy.signal.oaconvolve
    This is generally much faster than convolve for large arrays (n > ~500),
    and generally much faster than fftconvolve when one array is much larger
    than the other, but can be slower when only a few output values are needed
    or when the arrays are very similar in shape, and can only output float
    arrays (int or object array inputs will be cast to float).
    '''
    return signal.oaconvolve(arr1, arr2)
