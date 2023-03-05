#!/usr/bin/env python3

'''=================================================================
Author      : Jeffrey Allen
Class       : CS613, General Purpose GPU Programming
Assignment  : Semester Project
Description : Reference implementations of convolution tequniques
================================================================='''

''' Import downloaded dependencies '''
import numpy as np

''' Import built-in dependencies '''
from os import path, chdir
from math import floor

''' Import my defined dependencies '''
from convolve_funcs import (
    convolve_time_domain,
    convolve_fast_fourier_transform 
)
from utils import *

''' Get the show on the road '''
def main():

    # Define paths to testing data
    gold_test_dir = path.abspath(path.join(path.dirname(__file__),
        '../test_data_gold')
    )
    test_dir = path.abspath(path.join(path.dirname(__file__),
        '../test_data')
    )
    
    # Define setup info for run
    TESTING = False
    CONV_METHOD = "time" # "time" or "fft" or "overadd" or "oversave"
    NUM_ELEMENTS = 8192 # 1024, 2048, 4096, 8192, 16384, 32768, 65536

    if TESTING:
        signal = [8,2,5,4,1,7,3]
        filter = [1,3,5,3,1]
        filename_result = "result_frombook.txt"
    else:
        # Define paths for output data
        chdir(gold_test_dir)
        filename_1 = "arr1_" + str(NUM_ELEMENTS) + ".txt"
        filename_2 = "arr2_" + str(NUM_ELEMENTS) + ".txt"
        filename_result = CONV_METHOD + "_result_" + str(NUM_ELEMENTS) + ".txt"

        # Create data set if it doesn't exist
        if not path.exists(filename_1) or not path.exists(filename_2):
            create_random_data_set(filename_1, NUM_ELEMENTS)
            create_random_data_set(filename_2, NUM_ELEMENTS)

        # Read data set
        signal = read_array_from_text_file(filename_1)
        filter = read_array_from_text_file(filename_2)

    # Convolve
    match CONV_METHOD:
        case "time":
            r = floor(len(filter)/2)
            result = convolve_time_domain(signal, filter, r)
        case "fft":
            result = convolve_fast_fourier_transform(signal, filter)
    
    # Save result to file
    chdir(test_dir)
    save_array_to_text_file(filename_result, result)

    # Check if result is correct
    chdir(gold_test_dir)
    gold_test_data = read_array_from_text_file(filename_result)
    if np.allclose(result, gold_test_data):
        print("Result is correct")
    else:
        print("Incorrect result")

if __name__ == '__main__':
    main()