import pennylane as qml
import pickle
import numpy as np
import itertools
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


def load_circuits(file_path):
    """
    Load random circuits from a pkl file.
    """
    with open(file_path, 'rb') as f:
        circuits = pickle.load(f)
    return circuits

# Function to generate classical shadows and compute expectation values
def compute_classical_shadow_expectation(circuit_func, wire_labels, observable, num_samples=100):
    """
    Compute the expectation value of a local Pauli observable using classical shadows.
    
    Parameters:
    - circuit_func: A function representing the quantum circuit.
    - wire_labels: A list of wire labels for the device.
    - observable: A PennyLane observable, e.g., qml.PauliX(0) or qml.PauliZ(1).
    - num_samples: Number of samples to generate using classical shadows.
    
    Returns:
    - The estimated expectation value of the observable.
    """
    dev = qml.device("default.qubit", wires=wire_labels, shots=num_samples)

    @qml.qnode(dev)
    def shadow_node():
        circuit_func()  # Execute the circuit function
        return qml.classical_shadow(wires=wire_labels)
    
    # Generate classical shadow samples
    bits, recipes = shadow_node()
    shadow = qml.ClassicalShadow(bits, recipes)
    # Estimate the expectation value of the given observable using the classical shadow
    estimated_expectation = shadow.expval(observable)
    
    return estimated_expectation

def generate_pauli_observables(n_qubits, max_qubits):
    """
    Generate a list of Pauli observables that act on at most max_qubits qubits.
    
    Parameters:
    - n_qubits: Total number of qubits in the system.
    - max_qubits: Maximum number of qubits on which each observable should act.
    
    Returns:
    - A list of Pauli observables.
    """
    # Define single-qubit Pauli matrices
    pauli_matrices = [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]
    observables = []

    # Generate all possible combinations of up to max_qubits Pauli matrices
    for num_acting_qubits in range(1, max_qubits + 1):
        for qubit_indices in itertools.combinations(range(n_qubits), num_acting_qubits):
            for pauli_types in itertools.product(pauli_matrices[1:], repeat=num_acting_qubits):  # Exclude Identity from the product
                pauli_ops = [qml.Identity(i) for i in range(n_qubits)]  # Start with all Identity operators

                # Assign the chosen Pauli matrices to their respective qubits
                for qubit, pauli in zip(qubit_indices, pauli_types):
                    pauli_ops[qubit] = pauli(qubit)

                # Create the tensor product of the operators
                observables.append(pauli_ops[0])
                for op in pauli_ops[1:]:
                    observables[-1] = observables[-1] @ op  # Tensor product of observables

    return observables


def save_updated_circuits(file_path, updated_circuits):
    """
    Save the updated circuits to a pkl file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(updated_circuits, f)


def qasm_to_dag(qasm_str):
    # Create a QuantumCircuit from the QASM string
    circuit = QuantumCircuit.from_qasm_str(qasm_str)
    
    # Convert the QuantumCircuit to a DAG
    dag = circuit_to_dag(circuit)
    return dag



if __name__ == "__main__":

    qubit_ranges = range(2, 6)
    gate_ranges = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]
    
    for num_qubit in qubit_ranges:
        for gate_range in tqdm(gate_ranges, desc="Gate Ranges", leave=False):
            filename = f"random_circuits/basis_rotations+cx_qubits_{num_qubit}_gates_{gate_range[0]}-{gate_range[1]}.pkl"
            # Load circuits from pkl file
            circuits = load_circuits(filename)
            if isinstance(circuits[0], dict):
            # Check if any column in circuits starts with "obs"
                has_obs_column = any(key.startswith("obs_") for circuit in circuits for key in circuit.keys())
                if has_obs_column:
                    print(f"Skipping {filename}: Observable columns already present")
                    continue
                print(f"Processing {filename}: Adding observable columns")

                # From qasm to pennylane circuits
                pennylane_circuits = [qml.from_qasm(circuit['qasm']) for circuit in circuits]
                dag_circuits = [qasm_to_dag(circuit['qasm']) for circuit in circuits]

                # Create set of observables whose expectation value on the circuits constitute our designed features 
                list_observables = generate_pauli_observables(n_qubits=num_qubit, max_qubits=2)

                for i, circuit_data in enumerate(tqdm(circuits, desc="Circuits", leave=False)):
                    circuit_func = pennylane_circuits[i]
                    circuit_data["dag"] = dag_circuits[i]
                    
                    # Extract wire labels directly from the QASM string or circuit data
                    wire_labels = list(range(circuit_data["num_qubits"]))  # Assumes qubits are indexed from 0 to num_qubits-1
                    
                    # Define a local Pauli observable, e.g., PauliZ on qubit 0
                    for observable in list_observables:
                        observable_name = f"obs_{observable}"
                        
                        # Compute the expectation value of the specified observable
                        estimated_expectation = compute_classical_shadow_expectation(
                            circuit_func, wire_labels=wire_labels, observable=observable
                        )
                        # Add the expectation value to the circuit's data dictionary with the descriptive column name
                        circuit_data[observable_name] = estimated_expectation
            
            # Save the updated circuit data to a new pickle file
            save_updated_circuits(filename, circuits)
        
print(f"Dataset successfully generated and saved to {filename}")
