import pennylane as qml
import pickle
import os
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

def qasm_to_gate_counts(qasm_str):
    """
    Create a dictionary of gate counts and cumulative rotation angles from a QASM string.
    
    Returns:
    - Dictionary containing:
        - Gate counts for cx, h, rx, ry, rz gates
        - Cumulative rotation angles for rx, ry, rz gates per qubit
    """
    # Create a QuantumCircuit from the QASM string
    circuit = QuantumCircuit.from_qasm_str(qasm_str)
    num_qubits = circuit.num_qubits
    
    # Initialize counters for each gate type
    gate_counts = {
        "cx": 0,    # CNOT gate
        "h": 0,     # Hadamard
        "rx": 0,    # RX gate count
        "ry": 0,    # RY gate count
        "rz": 0     # RZ gate count
    }
    
    # Initialize rotation angle trackers for each qubit
    rotation_angles = {
        f"rx_q{i}": 0.0 for i in range(num_qubits)
    }
    rotation_angles.update({
        f"ry_q{i}": 0.0 for i in range(num_qubits)
    })
    rotation_angles.update({
        f"rz_q{i}": 0.0 for i in range(num_qubits)
    })
    
    # Count gates and track rotation angles
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        if gate_name in ["cx", "h"]:
            gate_counts[gate_name] += 1
        elif gate_name in ["rx", "ry", "rz"]:
            # Increment the gate count
            gate_counts[gate_name] += 1
            
            # Get the qubit object
            qubit = instruction.qubits[0]
            
            # Extract the index from the Qubit object
            try:
                qubit_idx = qubit.index
            except AttributeError:
                try:
                    qubit_idx = qubit._index
                except AttributeError:
                    qubit_str = str(qubit)
                    try:
                        qubit_idx = int(qubit_str.split(', ')[-1].split(')')[0])
                    except (ValueError, IndexError):
                        print(f"Warning: Using circuit position for {qubit}")
                        qubit_idx = 0
            
            # Get the parameter (angle) from the instruction
            angle = float(instruction.operation.params[0])
            # Add the absolute value of the angle to the cumulative sum
            rotation_angles[f"{gate_name}_q{qubit_idx}"] += abs(angle)
    
    # Combine gate counts and rotation angles into a single dictionary
    gate_counts.update(rotation_angles)
    
    return gate_counts

def qasm_to_gate_bins(qasm_str):
    """
    Create a dictionary of gate counts and binned rotation angles from a QASM string.
    The rotation angles (0, 2π) are divided into 50 bins for each rotation gate type.
    
    Returns:
    - Dictionary containing:
        - Gate counts for cx, h gates
        - Binned counts for rx, ry, rz gates (50 bins each)
    """
    # Create a QuantumCircuit from the QASM string
    circuit = QuantumCircuit.from_qasm_str(qasm_str)
    
    # Initialize counters for non-parameterized gates
    gate_counts = {
        "cx": 0,    # CNOT gate
        "h": 0,     # Hadamard
    }
    
    # Initialize bins for rotation gates
    num_bins = 50
    bin_edges = np.linspace(0, 2*np.pi, num_bins + 1)  # Create bin edges from 0 to 2π
    
    # Initialize histogram counts for each rotation gate type
    rotation_bins = {
        f"rx_bin_{i}": 0 for i in range(num_bins)
    }
    rotation_bins.update({
        f"ry_bin_{i}": 0 for i in range(num_bins)
    })
    rotation_bins.update({
        f"rz_bin_{i}": 0 for i in range(num_bins)
    })
    
    # Count gates and bin rotation angles
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        
        if gate_name in ["cx", "h"]:
            gate_counts[gate_name] += 1
            
        elif gate_name in ["rx", "ry", "rz"]:
            # Get the parameter (angle) from the instruction
            angle = float(instruction.operation.params[0])
            
            # Ensure angle is in [0, 2π)
            angle = angle % (2*np.pi)
            if angle < 0:
                angle += 2*np.pi
                
            # Find which bin this angle belongs to
            bin_idx = np.digitize(angle, bin_edges) - 1
            if bin_idx == num_bins:  # Handle edge case where angle = 2π
                bin_idx = num_bins - 1
                
            # Increment the appropriate bin counter
            rotation_bins[f"{gate_name}_bin_{bin_idx}"] += 1
    
    # Combine gate counts and rotation bins into a single dictionary
    gate_counts.update(rotation_bins)
    return gate_counts

def process_circuits(qubit_ranges=range(2, 7), gate_ranges=[(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]):
    for num_qubit in qubit_ranges:
        for gate_range in tqdm(gate_ranges, desc="Gate Ranges", leave=False):
            filename = f"dataset_random/basis_rotations+cx_qubits_{num_qubit}_gates_{gate_range[0]}-{gate_range[1]}.pkl"
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
                    
                    # Add gate counts to the circuit data
                    circuit_data["gate_counts"] = qasm_to_gate_counts(circuit_data['qasm'])
                    circuit_data["gate_bins"] = qasm_to_gate_bins(circuit_data['qasm'])
                    
                    # Extract wire labels directly from the QASM string or circuit data
                    wire_labels = list(range(circuit_data["num_qubits"]))
                    
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

def process_ising_circuits(min_qubit=2, max_qubit=7, min_trotter_step=10, max_trotter_step=11):
    qubit_ranges = range(min_qubit, max_qubit)  # Adjust as needed
    trotterization_steps_range = range(min_trotter_step, max_trotter_step)  # Adjust as needed
    
    for num_qubit in qubit_ranges:
        for trotter_steps in tqdm(trotterization_steps_range, desc="Trotterization Steps", leave=False):
            filename = f"dataset_tim/ising_qubits_{num_qubit}_trotter_{trotter_steps}.pkl"
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
                    
                    # Add gate counts to the circuit data
                    circuit_data["gate_counts"] = qasm_to_gate_counts(circuit_data['qasm'])
                    circuit_data["gate_bins"] = qasm_to_gate_bins(circuit_data['qasm'])
                    
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









if __name__ == "__main__":
    process_circuits()
    # process_ising_circuits()
    # gate_counts_to_circuits("tim", min_qubit=4, max_qubit=5, min_trotter_step=5, max_trotter_step=6)  # For tim dataset
    # gate_bins_to_circuits("random")

