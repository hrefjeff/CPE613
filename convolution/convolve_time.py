#!/usr/bin/env python3

import numpy as np

RADIUS = 2
FILTER_SIZE = 2*RADIUS + 1

def convolve(arr1, arr2):
    """ Convolve two arrays using the method from the book
        Page 153, section 7.1
    """
    # Create a new array to hold the result
    arr1_len = len(arr1)
    result = [0] * (arr1_len)

    # Convolve the two arrays
    for idxInput in range(arr1_len): # Loop over input array
        sum = 0
        for idxFilter in range(FILTER_SIZE): # Loop over convolution filter
            # Check if the filter element aligns with an input
            # element within the bounds of the input array
            inputPos = idxInput + (idxFilter - RADIUS)
            if inputPos < 0 or inputPos >= arr1_len:
                sum += 0
            else:
                sum += arr1[inputPos] * arr2[idxFilter]
        result[idxInput] = sum

    return result

def create_random_data_set(output_file, size=10**5):
    """Create a random data set and save to a text file"""
    arr = np.random.randint(0, 10, size)
    np.savetxt(output_file, arr)

def read_array_from_text_file(input_file):
    """Read a text file of numbers and return a numpy array"""
    arr = np.loadtxt(input_file)
    return arr

def main():
    
    TESTING = True

    if TESTING:
        arr1 = [8,2,5,4,1,7,3]
        arr2 = [1,3,5,3,1]
    else:
        # Create data set if it doesn't exist
        if not file_exists('arr1_time.txt') or not file_exists('arr2_time.txt'):
            create_random_data_set('arr1_time.txt')
            create_random_data_set('arr2_time.txt')
        # Read data set
        arr1 = read_array_from_text_file('arr1_time.txt')
        arr2 = read_array_from_text_file('arr2_time.txt')

    result = convolve(arr1, arr2)
    print(result)

if __name__ == '__main__':
    main()