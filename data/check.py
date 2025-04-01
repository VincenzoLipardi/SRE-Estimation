import os
import glob
import pickle

# Path to the specific file
file_path = "/data/P70087789/Shadows/data/dataset_tim/gate_counts/ising_qubits_4_trotter_4.pkl"

# Check if file exists
if os.path.exists(file_path):
    # Open and read the pickle file
    with open(file_path, 'rb') as f:  # Note: 'rb' for reading binary pickle files
        data = pickle.load(f)
        print("File contents:")

        for i in range(len(data)):
            print(len(data[i][0]))
else:
    print(f"File not found: {file_path}")
