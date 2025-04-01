import time
import numpy as np
import pandas as pd
import pickle  # Use pickle for loading models
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from models import load_data
from data.label import calculate_stabilizer_renyi_entropy_qiskit

def calculate_prediction_time(num_qubit, dataset="random", model_directory='experiments'):
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
    circuit_filename = f'random_circuits/basis_rotations+cx_qubits_{num_qubit}_gates_40-59.pkl'
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


# Initialize a list to store the results


# Loop through different numbers of qubits
def create_pkl(min_qubits, max_qubits):
    results = []
    for num_qubit in range(2, 6):
        prediction_time, entropy_time = calculate_prediction_time(num_qubit=num_qubit, dataset="random")
        results.append({'rf_time': prediction_time, 'sre_time': entropy_time, 'num_qubit': num_qubit})

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Save the DataFrame to a .pkl file
    df.to_pickle('experiments/prediction_entropy_times.pkl')
    print("File saved with times in milliseconds")
    
# Load the results from the .pkl file
results = pd.read_pickle('experiments/prediction_entropy_times.pkl')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.bar(results['num_qubit'], results['rf_time'], width=0.4, label='Random Forest Prediction Time (ms)', color='blue', align='center')
plt.bar(results['num_qubit'] + 0.4, results['sre_time'], width=0.4, label='Stabilizer Renyi Entropy Time (ms)', color='orange', align='center')

plt.xlabel('Number of Qubits')
plt.ylabel('Time (ms)')
plt.yscale('log')

plt.title('Prediction and Entropy Calculation Times')
plt.xticks(results['num_qubit'] + 0.2, results['num_qubit'])
plt.legend()
plt.grid(axis='y')

# Save the figure
plt.savefig('experiments/images/time_analysis.png', bbox_inches='tight', dpi=300)
plt.close()



