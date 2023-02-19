#!/usr/bin/env python3

import numpy as np
import scipy as sp
import timeit

def main():

    # Create random data set
    arr1 = np.random.random(100000)
    arr2 = np.random.random(100000)

    # Time the numpy normal convolve
    start = timeit.default_timer()
    np.convolve(arr1,arr2)
    end = timeit.default_timer() - start
    print("Numpy convolve time difference is :", end)

    # Time the scipy fftconvolve
    start = timeit.default_timer()
    sp.signal.fftconvolve(arr1,arr2)
    end = timeit.default_timer() - start
    print("Scipy fttconvolve time difference is :", end)

    # Check small data set that I know is correct
    # Should produce [4, 13, 28, 27, 18]
    print(np.convolve((1,2,3),(4,5,6)))
    print(sp.signal.fftconvolve((1,2,3),(4,5,6)))

if __name__ == '__main__':
    main()