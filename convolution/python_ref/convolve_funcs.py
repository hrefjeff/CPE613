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
