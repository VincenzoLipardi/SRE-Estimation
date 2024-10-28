import time
import numpy as np
import pickle  # Use pickle for loading models
from qiskit import QuantumCircuit
from random_forest import load_data
from data.label import calculate_stabilizer_renyi_entropy_qiskit

def calculate_prediction_time(num_qubit=2, dataset="random", model_directory='experiments'):
    # Directory where the data is stored
    directory = f'data/{dataset}_circuits'
    data = load_data(directory, num_qubit=num_qubit)[0]
    sampled = np.random.randint(0, 50000, size=2)
    random_instances = data[sampled]

    # Load the model using pickle
    model_path = f'{model_directory}/results_random_qubit_{num_qubit}.pkl'
    with open(model_path, 'rb') as file:
        results = pickle.load(file)

    model = results["Random_Forest"]
    # Measure the time for prediction
    start_time = time.perf_counter()
    model.predict(random_instances)
    end_time = time.perf_counter()

    # Calculate the duration in milliseconds
    prediction_time = (end_time - start_time) * 1000 / 2
    
    # Stabilizer Renyi entropy
    circuit_filename = f'data/random_circuits/basis_rotations+cx_qubits_{num_qubit}_gates_40-59.pkl'
    with open(circuit_filename, 'rb') as file:
        circuit_data = pickle.load(file)
    qasm_str = circuit_data[sampled[0] % 10000][0]['qasm']
    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
    # Measure the time for entropy calculation
    start_time_entropy = time.perf_counter()
    entropy = calculate_stabilizer_renyi_entropy_qiskit(qiskit_circuit)
    end_time_entropy = time.perf_counter()

    # Calculate the duration in milliseconds
    entropy_time = (end_time_entropy - start_time_entropy) * 1000
    return prediction_time, entropy_time


# Example usage
prediction_time, entropy_time = calculate_prediction_time(num_qubit=2, dataset="random")
print(f"Prediction Time: {prediction_time:.4f} ms")
print(f"Entropy Calculation Time: {entropy_time:.4f} ms")

