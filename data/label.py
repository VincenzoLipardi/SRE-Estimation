import pickle
import numpy as np
from itertools import product
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector


def calculate_stabilizer_renyi_entropy_qiskit(circuit, alpha=2):
    n = len(circuit.qubits)     # Number of qubits
    d = 2**n
    pauli_gates = ['I', 'X', 'Y', 'Z']
    gate_combinations = product(pauli_gates, repeat=n)      # All the pauli combination on all the qubits
    A = 0
    op = QuantumCircuit(n)

    for combination in gate_combinations:
        # Apply Pauli gates according to the combination
        for qubit, gate in enumerate(combination):
            if gate == 'X':
                op.x(qubit)
            elif gate == 'Y':
                op.y(qubit)
            elif gate == 'Z':
                op.z(qubit)
            else:
                pass

        # Calculate expectation value
        op = Operator(op)
        exp_val = Statevector(circuit).expectation_value(op).real
        A += ((1/d) * exp_val**2)**alpha
        # Recalculate the operator
        op = QuantumCircuit(n)

    entropy = (1 / (1 - alpha)) * np.log(A) - np.log(d)

    return entropy

def label_random_circuits(qubit_ranges=range(7, 8), gate_ranges=[(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]):
    for num_qubit in qubit_ranges:
        for gate_range in tqdm(gate_ranges, desc="Gate Ranges", leave=False):
            print("Qubit: ", num_qubit, "Gates: ", gate_range)
            filename = f"dataset_random/basis_rotations+cx_qubits_{num_qubit}_gates_{gate_range[0]}-{gate_range[1]}.pkl"

            with open(filename, 'rb') as file:
                data = pickle.load(file)
            
            # Check if the data is already processed
            if isinstance(data[0], tuple) and len(data[0]) == 2:
                print(f"File {filename} already processed. Skipping...")
                continue

            result = []

            num_qubits = data[0]['num_qubits']
            wire_labels = list(range(num_qubits))

            for circuit in tqdm(data, desc="Processing Circuits"):
                if isinstance(data[0], tuple):
                    qasm_str = circuit[0]['qasm']
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
                    # Create a list of tuples
                    result.append((circuit[0], calculate_stabilizer_renyi_entropy_qiskit(qiskit_circuit)))
                else:
                    qasm_str = circuit['qasm']
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
                    # Create a list of tuples
                    result.append((circuit, calculate_stabilizer_renyi_entropy_qiskit(qiskit_circuit)))
            # Save the updated data back to the original .pkl file
            # print(result)
            with open(filename, 'wb') as file:
                pickle.dump(result, file)
            print(f"Updated {len(result)} circuits with global expectation values and saved")

def label_ising_circuits(min_qubit=4, max_qubit=6, min_trotter_step=5, max_trotter_step=6):
    qubit_ranges = range(min_qubit, max_qubit)
    trotterization_steps_range = range(min_trotter_step, max_trotter_step)
    
    for num_qubit in qubit_ranges:
        for trotter_steps in tqdm(trotterization_steps_range, desc="Trotterization Steps", leave=False):
            filename = f"dataset_tim/ising_qubits_{num_qubit}_trotter_{trotter_steps}.pkl"
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            
            # Check if the data is already processed
            if isinstance(data[0], tuple) and len(data[0]) == 2:
                print(f"File {filename} already processed. Skipping...")
                continue

            result = []

            num_qubits = data[0]['num_qubits']
            wire_labels = list(range(num_qubits))

            for circuit in tqdm(data, desc="Processing Circuits"):
                if isinstance(data[0], tuple):
                    qasm_str = circuit[0]['qasm']
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
                    # Create a list of tuples
                    result.append((circuit[0], calculate_stabilizer_renyi_entropy_qiskit(qiskit_circuit)))
                else:
                    qasm_str = circuit['qasm']
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
                    # Create a list of tuples
                    result.append((circuit, calculate_stabilizer_renyi_entropy_qiskit(qiskit_circuit)))
            # Save the updated data back to the original .pkl file
            with open(filename, 'wb') as file:
                pickle.dump(result, file)
            print(f"Updated {len(result)} circuits with global expectation values and saved")

if __name__ == "__main__":
    # label_random_circuits()
    label_ising_circuits()
