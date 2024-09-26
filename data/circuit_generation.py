import pennylane as qml
from pennylane import numpy as np
import random
import pickle
import os
import qiskit
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

# Function to generate a random gate
def random_gate(num_qubits):
    gate_list = [
        qml.PauliX,   # X gate
        qml.PauliY,   # Y gate
        qml.PauliZ,   # Z gate
        qml.Hadamard, # H gate
        qml.S,        # S gate
        qml.T,        # T gate
        qml.RX,       # RX rotation
        qml.RY,       # RY rotation
        qml.RZ,       # RZ rotation
        qml.CNOT      # CNOT gate
    ]
    
    # Pick a random gate
    gate = random.choice(gate_list)
    
    # Generate a random qubit index
    qubit_index = random.randint(0, num_qubits - 1)
    
    # For rotation gates, also generate a random angle
    if gate in [qml.RX, qml.RY, qml.RZ]:
        return gate(np.random.uniform(0, 2 * np.pi), wires=qubit_index)
    elif gate == qml.CNOT:
        # For CNOT, choose two distinct qubits for control and target
        control = random.randint(0, num_qubits - 1)
        target = random.randint(0, num_qubits - 1)
        while target == control:
            target = random.randint(0, num_qubits - 1)
        return gate(wires=[control, target])
    else:
        return gate(wires=qubit_index)

# Function to generate a random quantum circuit and return the final state
def random_quantum_circuit(num_qubits, num_gates):
    # Define a device with the given number of qubits
    dev = qml.device('lightning.qubit', wires=num_qubits)

    # Define a quantum node (qnode)
    @qml.qnode(dev)
    def circuit():
        # Apply random gates
        for _ in range(num_gates):
            random_gate(num_qubits)
        return qml.state()  # Return the statevector of the quantum system

    # Construct the circuit
    circuit.construct([], {})
    
    # Convert to OpenQASM
    qasm_str = circuit.qtape.to_openqasm()
    
    # Optionally, you can remove the measurement from the QASM string
    # qasm_str = qasm_str[:-23]  # This would remove the final measurement line
    
    return qasm_str

# Function to generate random circuits within a range of qubits and gates and save them
def generate_and_save_circuits(qubit_range, gate_range, num_circuits):
    # Create a directory for saving the QASM files
    os.makedirs("data", exist_ok=True)
    
    circuits = []
    for i in range(num_circuits):
        # Randomly select the number of qubits and gates from the given ranges
        num_qubits = random.randint(qubit_range[0], qubit_range[1])
        num_gates = random.randint(gate_range[0], gate_range[1])
        
        # Generate a random quantum circuit and get the QASM string
        qasm_str = random_quantum_circuit(num_qubits, num_gates)
        circuits.append({
            'qasm': qasm_str,
            'num_qubits': num_qubits,
            'num_gates': num_gates
        })
    
    # Dynamically create the filename based on the ranges
    filename = f"random_circuits_qubits_{qubit_range[0]}-{qubit_range[1]}_gates_{gate_range[0]}-{gate_range[1]}.pkl"
    
    # Save the generated circuits to a .pkl file in the 'data' directory
    filepath = os.path.join('data', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(circuits, f)
    print(f"Saved {num_circuits} random quantum circuits to {filepath}")
    
    return filepath
def pennylane_circuit_from_qasm(qasm_str, num_qubits):
    dev = qml.device('default.qubit', wires=num_qubits)  # Assuming max 10 qubits, adjust if needed
    
    @qml.qnode(dev)
    def circuit():
        qml.from_qasm(qasm_str)
        return qml.state()
    #print(qml.draw(circuit)())
    #print("\n" + "-"*50 + "\n")
    return circuit

def qiskit_circuit_from_qasm(qasm_str, num_qubits):
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    return qc

# Example usage
qubit_range = (2, 5)  # Range of qubits
gate_range = (5, 15)  # Range of gates
num_circuits = 10     # Number of circuits to generate

# Generate and save random circuits
saved_filepath = generate_and_save_circuits(qubit_range, gate_range, num_circuits)

# Function to load the saved circuits
def load_circuits(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Load the saved circuits
loaded_circuits = load_circuits(saved_filepath)

# Print some information about the loaded circuits
print(f"Loaded {len(loaded_circuits)} circuits")
for i, circuit in enumerate(loaded_circuits[:]):  # Print info for the first 3 circuits
    print(f"Circuit {i+1}: {circuit['num_qubits']} qubits, {circuit['num_gates']} gates\n {qiskit_circuit_from_qasm(circuit['qasm'], circuit['num_qubits'])}")
    
