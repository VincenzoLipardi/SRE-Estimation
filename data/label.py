import pickle
import pennylane as qml
import numpy as np
from itertools import product
from tqdm import tqdm

def generate_global_observable(num_qubits):
    """
    Generate a global observable for the given number of qubits.
    This function creates a tensor product of Pauli Z operators for all qubits.
    """
    observable = qml.PauliZ(0)
    for i in range(1, num_qubits):
        observable = observable @ qml.PauliZ(i)  # Ensure all qubits are included
    return observable

def calculate_global_expectation(wire_labels, circuit, observable):
    """
    Calculate the expectation value of a global observable for a given quantum circuit.

    Args:
    wire_labels (list): List of wire labels for the device.
    qasm_str (str): QASM string representation of the quantum circuit.

    Returns:
    float: Expectation value of the global observable.
    """
    dev = qml.device("default.qubit", wires=wire_labels, shots=100)
    
    @qml.qnode(dev)
    def shadow_node():
        circuit()  # Execute the circuit function
        return qml.classical_shadow(wires=wire_labels)
    
    # Generate classical shadow samples
    bits, recipes = shadow_node()
    shadow = qml.ClassicalShadow(bits, recipes)
    # Estimate the expectation value of the given observable using the classical shadow
    
    estimated_expectation = shadow.expval(observable)
    return estimated_expectation

def calculate_von_neumann_entropy(wire_labels, circuit):
    """
    Calculate the von Neumann entropy for a given quantum circuit.

    Args:
    wire_labels (list): List of wire labels for the device.
    circuit (callable): Quantum circuit function.

    Returns:
    float: Von Neumann entropy of the quantum state.
    """
    dev = qml.device("default.qubit", wires=wire_labels)
    
    @qml.qnode(dev)
    def entropy_node():
        circuit()  # Execute the circuit function
        return qml.state()  # Return the quantum state
    
    # Get the quantum state
    state = entropy_node()
    density_matrix = qml.math.outer(state, qml.math.conj(state))
    
    # Calculate the von Neumann entropy
    entropy = qml.math.vn_entropy(density_matrix, indices=[0])
    return entropy



def calculate_stabilizer_renyi_entropy(wire_labels, pennylane_circuit, alpha=2):
    # Set up the PennyLane device
    n = len(wire_labels)
    d = 2**n
    pauli_gates = ['I', 'X', 'Y', 'Z']
    gate_combinations = list(product(pauli_gates, repeat=n))
    A = 0

    # Define a helper function to apply the Pauli gate combination on the circuit
    def apply_pauli_combination(combination):
        for qubit, gate in enumerate(combination):
            if gate == 'X':
                qml.PauliX(wires=qubit)
            elif gate == 'Y':
                qml.PauliY(wires=qubit)
            elif gate == 'Z':
                qml.PauliZ(wires=qubit)
            # 'I' gate is just identity, so nothing needs to be applied for it

    # Create a device for each combination to calculate the expectation value
    for combination in gate_combinations:
        # Create a QNode for the current combination
        @qml.qnode(qml.device('default.qubit', wires=wire_labels))
        def circuit_with_pauli():
            pennylane_circuit()  # Apply the original circuit
            apply_pauli_combination(combination)  # Apply Pauli gates based on the combination
            return qml.expval(qml.PauliZ(0))  # Placeholder, all Pauli Z expectations can be computed here

        # Calculate expectation value and add to A
        exp_val = circuit_with_pauli()
        A += ((1 / d) * exp_val ** 2) ** alpha

    entropy = (1 / (1 - alpha)) * np.log(A) - np.log(d)
    return entropy


if __name__ == "__main__":
    qubit_ranges = range(2,4)
    gate_ranges = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]
    
    for num_qubit in qubit_ranges:
        for gate_range in tqdm(gate_ranges, desc="Gate Ranges", leave=False):
            filename = f"data/random_circuits/basis_rotations+cx_qubits_{num_qubit}_gates_{gate_range[0]}-{gate_range[1]}.pkl"

            with open(filename, 'rb') as file:
                data = pickle.load(file)

            # Calculate global expectation for each circuit
            result = []
            num_qubits = data[0][0]['num_qubits']
            wire_labels = list(range(num_qubits))
            if isinstance(data[0], tuple):
                if len(data[0])== 1:
                    print("tiple lenght 1")
                    for circuit in tqdm(data, desc="Processing Circuits"):
                        qasm_str = circuit[0]['qasm']
                        pennylane_circuit = qml.from_qasm(qasm_str)
                        # Create a list of tuples
                        result.append((circuit[0], calculate_stabilizer_renyi_entropy(wire_labels, pennylane_circuit)))
                elif len(data[0])== 2 or len(data[0])==3:
                    for circuit in tqdm(data, desc="Processing Circuits"):
                        qasm_str = circuit[0]['qasm']
                        pennylane_circuit = qml.from_qasm(qasm_str)
                        # Create a list of tuples
                        result.append((circuit[0], calculate_stabilizer_renyi_entropy(wire_labels, pennylane_circuit)))
                else:
                    print("Data is weird")
            else:
                for circuit in tqdm(data, desc="Processing Circuits"):
                    qasm_str = circuit['qasm']
                    pennylane_circuit = qml.from_qasm(qasm_str)
                    # Create a list of tuples
                    result.append((circuit, calculate_stabilizer_renyi_entropy(wire_labels, pennylane_circuit)))
            print(len(result), len(result[0]))
            # Save the updated data back to the original .pkl file
            with open(filename, 'wb') as file:
                pickle.dump(result, file)
            print(f"Updated {len(result)} circuits with global expectation values and saved")
