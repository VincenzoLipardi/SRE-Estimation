import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the directory containing the files
directory = 'experiments'

# List to store dataframes
dataframes = []

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .pkl file and contains "results" in its name
    if filename.endswith('.pkl') and 'results' in filename:
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Load the dataframe from the .pkl file
        with open(file_path, 'rb') as file:
            df = pd.read_pickle(file)
            dataframes.append(df)
print(dataframes[0]["Dataset"])


