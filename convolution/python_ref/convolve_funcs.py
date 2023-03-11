import numpy as np
from scipy import signal

def convolve_time_domain(arr1, arr2, radius):
    """ Naive convolve method that convoles arr2 onto arr1 """

    # Sets up arrays to work with
    arr1 = np.pad(arr1, radius, 'constant') # Pad array with 0's
    arr2 = np.flip(arr2) # If we don't flip array, it's just a correlation
    arr1_len = len(arr1)
    arr2_len = len(arr2)
    result = [0] * (arr1_len)

    for idxInput in range(arr1_len): # Loop over input array
        startPos = idxInput - radius
        sum = 0
        for idxFilter in range(arr2_len): # Loop over convolution filter
            # Check if the positions to add are within the bounds
            if (startPos + idxFilter) < 0 or (startPos + idxFilter) >= arr1_len:
                sum += 0
            else:
                sum += arr1[startPos + idxFilter] * arr2[idxFilter]
        result[idxInput] = sum
        
    return result

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
