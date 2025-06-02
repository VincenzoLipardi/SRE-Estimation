import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from training import load_data
from data.label import calculate_stabilizer_renyi_entropy_qiskit


def calculate_training_time(num_qubit, dataset="random", model="rfr"):
    # Load data
    directory = f'data/dataset_{dataset}/gate_bins'
    data, labels = load_data(directory, num_qubit=num_qubit, gate_filter_value=False)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create model
    if model == "rfr":
        model_instance = RandomForestRegressor(n_estimators=300, max_features='sqrt', criterion='squared_error', random_state=42)
    elif model == "svr":
        model_instance = SVR(kernel='rbf', C=1, epsilon=0.1)
    else:
        raise ValueError("Unknown model")

    # Measure training time
    start_train = time.perf_counter()
    model_instance.fit(X_train, y_train)
    end_train = time.perf_counter()
    training_time = (end_train - start_train) * 1000  # ms

    # Measure prediction time (on X_test)
    start_pred = time.perf_counter()
    _ = model_instance.predict(X_test)
    end_pred = time.perf_counter()
    prediction_time = (end_pred - start_pred) * 1000  # ms

    # Classical SRE time (single circuit)
    circuit_filename = f'data/dataset_{dataset}/basis_rotations+cx_qubits_{num_qubit}_gates_40-59.pkl'
    with open(circuit_filename, 'rb') as file:
        circuit_data = pickle.load(file)
    # Compute average SRE time over 50 random circuits in the file
    num_samples = min(50, len(circuit_data))
    sampled_indices = np.random.choice(len(circuit_data), size=num_samples, replace=False)
    entropy_times = []
    for idx in sampled_indices:
        qasm_str = circuit_data[idx][0]['qasm']
        qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
        start_time_entropy = time.perf_counter()
        _ = calculate_stabilizer_renyi_entropy_qiskit(qiskit_circuit)
        end_time_entropy = time.perf_counter()
        entropy_times.append((end_time_entropy - start_time_entropy) * 1000)
    entropy_time = np.mean(entropy_times)

    return training_time, prediction_time, entropy_time


def create_pkl(min_qubits, max_qubits):
    results = []
    for num_qubit in range(min_qubits, max_qubits):
        # RFR
        rfr_train, rfr_pred, entropy_time = calculate_training_time(num_qubit=num_qubit, dataset="random", model="rfr")
        # SVR
        svr_train, svr_pred, _ = calculate_training_time(num_qubit=num_qubit, dataset="random", model="svr")
        results.append({
            'num_qubit': num_qubit,
            'rfr_train': rfr_train,
            'rfr_pred': rfr_pred,
            'svr_train': svr_train,
            'svr_pred': svr_pred,
            'sre_time': entropy_time
        })
    df = pd.DataFrame(results)
    df.to_pickle('experiments/training_entropy_times.pkl')
    print("File saved with times in milliseconds")

# Run for both models
# create_pkl(2, 7)

# Load the results from the .pkl file
results = pd.read_pickle('experiments/training_entropy_times.pkl')

# Print the running times in a readable format
print('Number of Qubits | RFR Train (ms) | RFR Pred (ms) | SVR Train (ms) | SVR Pred (ms) | SRE Time (ms)')
for idx, row in results.iterrows():
    print(f"{row['num_qubit']:>15} | {row['rfr_train']:>13.2f} | {row['rfr_pred']:>12.2f} | {row['svr_train']:>13.2f} | {row['svr_pred']:>12.2f} | {row['sre_time']:>12.2f}")

# Plotting the results
plt.figure(figsize=(12, 7))
bar_width = 0.25
x = results['num_qubit']

# RFR stacked
plt.bar(x - bar_width, results['rfr_train'], width=bar_width, label='RFR Training Time (ms)', color="#F0E442", align='center')
plt.bar(x - bar_width, results['rfr_pred'], width=bar_width, bottom=results['rfr_train'], label='RFR Prediction Time (ms)', color="#56B4E9", align='center')
# SVR stacked
plt.bar(x, results['svr_train'], width=bar_width, label='SVR Training Time (ms)', color="#E69F00", align='center')
plt.bar(x, results['svr_pred'], width=bar_width, bottom=results['svr_train'], label='SVR Prediction Time (ms)', color='#009E73', align='center')
# SRE time as separate bar
plt.bar(x + bar_width, results['sre_time'], width=bar_width, label='SRE Computation Time (ms)', color="#0072B2", align='center')

plt.xlabel('Number of qubits', fontsize=18)
plt.ylabel('Time (ms)', fontsize=18)
plt.yscale('log')
# plt.title('Training, Prediction, and Entropy Calculation Times for RFR and SVR', fontsize=20)
plt.xticks(x, results['num_qubit'], fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="best", fontsize=16)
plt.grid(axis='y')
plt.savefig('experiments/images/time_analysis.png', bbox_inches='tight', dpi=300)
plt.close()

