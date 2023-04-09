#!/usr/bin/env python3

from os import path, chdir
import numpy as np
from utils import read_array_from_text_file

gold_file = 'time_result_1024.txt'
result_file = 'cuda_time_1024.txt'

# Define paths to testing data
gold_test_dir = path.abspath(path.join(path.dirname(__file__),
    '../test_data_gold')
)
test_dir = path.abspath(path.join(path.dirname(__file__),
    '../test_data')
)

# Check if result is correct
chdir(gold_test_dir)
gold_test_data = read_array_from_text_file(gold_file)
chdir(test_dir)
test_data = read_array_from_text_file(result_file)

if np.allclose(test_data, gold_test_data):
    print("Result is correct")
else:
    print("Incorrect result")