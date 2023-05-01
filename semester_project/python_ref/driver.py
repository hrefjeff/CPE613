#!/usr/bin/env python3

'''=================================================================
Author      : Jeffrey Allen
Class       : CS613, General Purpose GPU Programming
Assignment  : Semester Project
Description : Reference implementations of convolution techniques
================================================================='''

''' Import built-in dependencies '''
from os import path, chdir
import timeit

''' Import my defined dependencies '''
from convolve_funcs import (
    convolve_time_domain,
    convolve_time_domain_np,
    convolve_fft,
    convolve_fft_sp
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
    CONV_METHOD = "fft" # "time", "np_time", "fft", "sp_fft" "overadd", "oversave"
    NUM_ELEMENTS = 1024 # 1024, 2048, 4096, 8192, 16384, 32768, 65536

    if TESTING:
        ''' Testing data comes from the book "The Scientist and Engineer's
            Guide to Digital Signal Processing" on page 116'''
        signal = [0, -1, -1.2, 2, 1.4, 1.4, 0.6, 0]
        filter = [1, -.5, -.25, -.1]
        filename_result = "result_testdata.txt"
    else:
        # Define paths for output data
        chdir(gold_test_dir)
        filename_1 = "arr1_" + str(NUM_ELEMENTS) + ".txt"
        filename_2 = "arr2_" + str(NUM_ELEMENTS) + ".txt"
        filename_result = "py_" + CONV_METHOD + "_result_" + \
            str(NUM_ELEMENTS) + ".txt"

        # Create data set if it doesn't exist
        if not path.exists(filename_1) or not path.exists(filename_2):
            create_random_data_set(filename_1, NUM_ELEMENTS)
            create_random_data_set(filename_2, NUM_ELEMENTS)

        # Read data set
        signal = read_array_from_text_file(filename_1)
        filter = read_array_from_text_file(filename_2)

    # Convolve

    start = timeit.default_timer()
    match CONV_METHOD:
        case "time":
            result = convolve_time_domain(signal, filter)
        case "np_time":
            result = convolve_time_domain_np(signal, filter)
        case "fft":
            result = convolve_fft(signal, filter)
        case "sp_fft":
            result = convolve_fft_sp(signal, filter)
    end = timeit.default_timer() - start
    
    #print(f'Python convolution time is {end*1000:.1f} seconds')
    print(f'Python convolution time is {end:.1f} milliseconds')

    # Save result to file
    chdir(test_dir)
    save_array_to_text_file(filename_result, result)
    # plot_result(result)
    
    # Check if result is correct
    chdir(gold_test_dir)
    gold_test_data = read_array_from_text_file(f'time_result_{NUM_ELEMENTS}.txt')
    if np.allclose(result, gold_test_data):
        print("Result is correct")
    else:
        print("Incorrect result")

if __name__ == '__main__':
    main()