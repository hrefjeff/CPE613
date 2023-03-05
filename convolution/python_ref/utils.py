import numpy as np

def create_random_data_set(output_file, size=10**5):
    """Create a random data set and save to a text file"""
    arr = np.random.randint(0, 100, size)
    np.savetxt(output_file, arr)

def read_array_from_text_file(input_file):
    """Read a text file of numbers and return a numpy array"""
    arr = np.loadtxt(input_file)
    return arr

def save_array_to_text_file(output_file, arr):
    """Save a numpy array to a text file"""
    np.savetxt(output_file, arr, fmt='%d')