#!/usr/bin/env python3

import numpy as np
import scipy as sp
import timeit

def file_exists(file_name):
    """Check if a text file exists"""
    try:
        with open(file_name, 'r') as f:
            pass
    except FileNotFoundError:
        return False
    return True

def create_random_data_set(output_file, size=10**5):
    """Create a random data set and save to a text file"""
    arr = np.random.randint(0, 10, size)
    np.savetxt(output_file, arr)

def read_array_from_text_file(input_file):
    """Read a text file of numbers and return a numpy array"""
    arr = np.loadtxt(input_file)
    return arr

def main():

    # Create data set if it doesn't exist
    if not file_exists('arr1.txt') or not file_exists('arr2.txt'):
        create_random_data_set('arr1.txt')
        create_random_data_set('arr2.txt')
    
    # Read data set
    arr1 = read_array_from_text_file('arr1.txt')
    arr2 = read_array_from_text_file('arr2.txt')

    # Time the numpy normal convolve
    start = timeit.default_timer()
    np.convolve(arr1,arr2)
    end = timeit.default_timer() - start
    print(f'Numpy convolve time is {end:.1f} ms')

    # Time the scipy fftconvolve
    start = timeit.default_timer()
    sp.signal.fftconvolve(arr1,arr2)
    end = timeit.default_timer() - start
    print(f'Scipy fttconvolve time is {end:.1f} ms')

    # Check small data set that I know is correct
    # Should produce [4, 13, 28, 27, 18]
    print(np.convolve((1,2,3),(4,5,6)))
    print(sp.signal.fftconvolve((1,2,3),(4,5,6)))

if __name__ == '__main__':
    main()