''' Import downloaded dependencies '''
import numpy as np
import matplotlib.pyplot as plt


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
    #np.savetxt(output_file, arr, fmt='%20.16e') # scientific notation
    np.savetxt(output_file, arr, fmt='%20.16f')  # decimal notation

def plot_result(result):
    # Plot result
    x = np.linspace(0, len(result), len(result))
    plt.plot(x, result)
    plt.title('Sample plot')
    plt.xlabel('Index')
    plt.savefig("mygraph.png")