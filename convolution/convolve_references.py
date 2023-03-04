#!/usr/bin/env python3

'''=================================================================
Author      : Jeffrey Allen
Class       : CS613, General Purpose GPU Programming
Assignment  : Semester Project
Description : Time domain convolution reference implementation
================================================================='''

import numpy as np

from math import floor
from os.path import exists

from utils import *

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

def main():
    
    TESTING = False

    if TESTING:
        signal = [8,2,5,4,1,7,3]
        filter = [1,3,5,3,1]
    else:
        # Create data set if it doesn't exist
        if not exists('arr1_1k.txt') or not exists('arr2_1k.txt'):
            create_random_data_set('arr1_1k.txt',10**4)
            create_random_data_set('arr2_1k.txt',10**4)
        # Read data set
        signal = read_array_from_text_file('arr1_1k.txt')
        filter = read_array_from_text_file('arr2_1k.txt')

    r = floor(len(filter)/2)
    result = convolve_time_domain(signal, filter, r)
    save_array_to_text_file('result.txt', result)

if __name__ == '__main__':
    main()