import pennylane as qml
from pennylane import numpy as np
import random
import pickle
import os
from qiskit import QuantumCircuit

# Function to generate a random gate
def random_gate(num_qubits, basis_gates):
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
    if basis_gates == "Clifford+T" or basis_gates == "clifford+t" or basis_gates == "Clifford+t" or basis_gates == "clifford+T":
        gate_list = [
            qml.PauliX,   # X gate
            qml.PauliY,   # Y gate
            qml.PauliZ,   # Z gate
            qml.Hadamard, # H gate
            qml.S,        # S gate
            qml.CNOT,     # CNOT gate
            qml.T,        # T gate
        ]
    elif basis_gates == "Clifford" or basis_gates == "clifford":
        gate_list = [
            qml.Hadamard, # H gate
            qml.S,        # S gate
            qml.T,        # T gate
            qml.RX,       # RX rotation
            qml.RY,       # RY rotation
            qml.RZ,       # RZ rotation
            qml.CNOT      # CNOT gate
        ]
    elif basis_gates == "rotations+cx" or basis_gates == "rotations+cnot" or basis_gates == "Rotations+cx" or basis_gates == "Rotations+cnot" or basis_gates == "Rotations+CNOT" or basis_gates == "Rotations+CNOT":
        gate_list = [
            qml.RX,       # RX rotation
            qml.RY,       # RY rotation
            qml.RZ,       # RZ rotation
            qml.CNOT      # CNOT gate
        ]
    else:
        raise ValueError(f"Invalid basis gates: {basis_gates}")
    
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
def random_quantum_circuit(num_qubits, num_gates, basis_gates):
    # Define a device with the given number of qubits
    dev = qml.device('lightning.qubit', wires=num_qubits)

    # Define a quantum node (qnode)
    @qml.qnode(dev)
    def circuit():
        # Apply random gates
        for _ in range(num_gates):
            random_gate(num_qubits, basis_gates)
        return qml.state()  # Return the statevector of the quantum system

    # Construct the circuit
    circuit.construct([], {})
    
    # Convert to OpenQASM
    qasm_str = circuit.qtape.to_openqasm()
    
    # Remove the measurement and classical bits from the QASM string
    qasm_lines = qasm_str.split('\n')
    qasm_str = '\n'.join([line for line in qasm_lines if not line.startswith('creg') and not line.startswith('measure')])
    
    return qasm_str

# Function to generate random circuits within a range of qubits and gates and save them
def generate_and_save_circuits(num_qubits, gate_range, num_circuits, basis_gates):
    # Create a directory for saving the QASM files
    os.makedirs("data", exist_ok=True)
    
    circuits = []
    for i in range(num_circuits):
        # Randomly select the number of qubits and gates from the given ranges

        num_gates = random.randint(gate_range[0], gate_range[1])
        
        # Generate a random quantum circuit and get the QASM string
        qasm_str = random_quantum_circuit(num_qubits, num_gates, basis_gates)
        circuits.append({
            'qasm': qasm_str,
            'num_qubits': num_qubits,
            'num_gates': num_gates
        })
    
    # Dynamically create the filename based on the ranges
    filename = f"random_circuits_basis_{basis_gates}_qubits_{num_qubits}_gates_{gate_range[0]}-{gate_range[1]-1}.pkl"
    
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

# Data generatio
num_circuits = 10000     # Number of circuits to generate
basis_gates = "rotations+cx"
# Generate and save random circuits
for i in range(2,11):
    for j in range(4,5):
        saved_filepath = generate_and_save_circuits(num_qubits=i, gate_range=(20*j, 20*(j+1)), num_circuits=num_circuits, basis_gates=basis_gates)
