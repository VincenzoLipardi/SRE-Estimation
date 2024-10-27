import pickle
from qiskit.visualization import dag_drawer

# Define the file path
file_path = 'data/random_circuits/basis_rotations+cx_qubits_4_gates_0-19.pkl'

# Open the file in binary read mode and load the data
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Print the data to verify
print(len(data), len(data[1]))
dag_drawer(data[1]["dag"], filename= "data/random_circuits/images/dag.png")
