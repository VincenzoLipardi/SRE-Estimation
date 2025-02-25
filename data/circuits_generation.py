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

def trotterized_ising_circuits(num_qubits, trotterization_steps, rx_angle):
    # Define a device with the given number of qubits
    dev = qml.device('lightning.qubit', wires=num_qubits)

    # Define a quantum node (qnode)
    @qml.qnode(dev)
    def circuit():
        # Initialize the qubits in the |+> state
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        
        # Apply Trotterized Ising model evolution
        for _ in range(trotterization_steps):
            # Apply ZZ interactions
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                # Ensure RZ angle is less than RX angle
                rz_angle = np.random.uniform(0, rx_angle)
                qml.RZ(rz_angle, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            
            # Apply local X rotations with the input RX angle
            for i in range(num_qubits):
                qml.RX(rx_angle, wires=i)

        return qml.state()  # Return the statevector of the quantum system

    # Construct the circuit
    circuit.construct([], {})
    
    # Convert to OpenQASM
    qasm_str = circuit.qtape.to_openqasm()
    
    # Remove the measurement and classical bits from the QASM string
    qasm_lines = qasm_str.split('\n')
    qasm_str = '\n'.join([line for line in qasm_lines if not line.startswith('creg') and not line.startswith('measure')])
    
    return qasm_str

# Function to generate random circuits or Ising model circuits within a range of qubits and gates and save them
def generate_and_save_circuits(directory, num_qubits, num_circuits, basis_gates, circuit_type='random', gate_range=None, trotterization_steps=1, rx_angle=0.1):
    # Create a directory for saving the QASM files
    os.makedirs(directory, exist_ok=True)
    
    circuits = []
    if circuit_type == 'random':
        if gate_range is None:
            raise ValueError("gate_range must be provided for random circuits")
        for _ in range(num_circuits):
            # Randomly select the number of gates from the given range
            num_gates = random.randint(gate_range[0], gate_range[1])
            # Generate the random quantum circuit and get the QASM string
            qasm_str = random_quantum_circuit(num_qubits, num_gates, basis_gates)
            circuits.append({
                'qasm': qasm_str,
                'num_qubits': num_qubits,
                'num_gates': num_gates
            })
    elif circuit_type == 'ising':
        for _ in range(num_circuits):
            # Generate a random RX angle in the range [0, 2π]
            random_rx_angle = np.random.uniform(0, 2 * np.pi)
            # Generate the Trotterized Ising circuit and get the QASM string
            qasm_str = trotterized_ising_circuits(num_qubits, trotterization_steps, random_rx_angle)
            circuits.append({
                'qasm': qasm_str,
                'num_qubits': num_qubits,
                'h': 'random',  # Use 'random' to indicate RX angles are random
                'J': 'random'   # Placeholder for RZ angle, as it is random
            })
    else:
        raise ValueError(f"Invalid circuit type: {circuit_type}")

    # Define the filename based on the circuit type
    if circuit_type == 'random':
        filename = f"{circuit_type}_basis_{basis_gates}_qubits_{num_qubits}_gates_{gate_range[0]}-{gate_range[1]-1}.pkl"
    elif circuit_type == 'ising':
        filename = f"ising_qubits_{num_qubits}_trotter_{trotterization_steps}.pkl"

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    filepath = os.path.join(directory, filename)
    
    # Check if the file already exists
    if os.path.exists(filepath):
        print(f"File {filepath} already exists")
        return filepath
    
    with open(filepath, 'wb') as f:
        pickle.dump(circuits, f)
    print(f"Saved {num_circuits} {circuit_type} quantum circuits to {filepath}")
    
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

def generate_random_circuits():
    random.seed(0)
    random_directory = "dataset_random"
    random_num_circuits = 10000
    random_basis_gates = "rotations+cx"
    ranges = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]

    for start, end in ranges:
        random_gate_range = (start, end)
        filename_suffix = f"{start}-{end}"
        for i in range(2, 7):  # Loop over the number of qubits
            saved_filepath_random = generate_and_save_circuits(
                random_directory,
                num_qubits=i,
                num_circuits=random_num_circuits,
                basis_gates=random_basis_gates,
                circuit_type='random',
                gate_range=random_gate_range
            )

def generate_ising_circuits():
    ising_directory = "dataset_tim"
    ising_num_circuits = 1000
    ising_rx_angle = 0.1

    for i in range(2, 7):
        for j in range(1, 6):
            saved_filepath_ising = generate_and_save_circuits(
                ising_directory,
                num_qubits=i,
                num_circuits=ising_num_circuits,
                basis_gates="rotations+cx",  # Assuming same basis for consistency
                circuit_type='ising',
                trotterization_steps=j,
                rx_angle=ising_rx_angle
            )

if __name__ == "__main__":
    # Choose which type of circuits to generate
    #generate_random_circuits()
    generate_ising_circuits()



