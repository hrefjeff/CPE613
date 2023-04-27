#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Generate a range of N values
N_values = np.arange(1, 500)

# Calculate O(N^2) and O(N*log(N)) for each N value
O_N_squared = N_values ** 2
O_N_logN = N_values * np.log(N_values)

# Create a line graph
plt.plot(N_values, O_N_squared, label="O(N^2)")
plt.plot(N_values, O_N_logN, label="O(N*log(N))")

# Set axis labels and title
plt.xlabel("Number of Elements (N)")
plt.ylabel("Runtime (In Millions)")
plt.title("Comparison of O(N^2) and O(N*log(N)) Runtimes")

# Add a legend
plt.legend()

# Display the graph
plt.savefig("runtime_comparison.png", dpi=350)