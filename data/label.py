import pickle
import pennylane as qml
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

if __name__ == "__main__":
    qubit_ranges = range(2,6)
    gate_ranges = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]
    
    for num_qubit in qubit_ranges:
        for gate_range in tqdm(gate_ranges, desc="Gate Ranges", leave=False):
            filename = f"random_circuits/basis_rotations+cx_qubits_{num_qubit}_gates_{gate_range[0]}-{gate_range[1]}.pkl"

            with open(filename, 'rb') as file:
                data = pickle.load(file)

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
            print(result)
            with open(filename, 'wb') as file:
                pickle.dump(result, file)
            print(f"Updated {len(result)} circuits with global expectation values and saved")
