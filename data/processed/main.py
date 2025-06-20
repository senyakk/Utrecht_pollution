"""
The code below checks ranges of values in a dataset
"""

import pandas as pd

data = pd.read_csv("forecast_example.csv")

min_values = data.min()
max_values = data.max()

print("Minimum values for each column:")
print(min_values)

print("\nMaximum values for each column:")
print(max_values)
