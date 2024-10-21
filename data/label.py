import pickle
import pennylane as qml
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

if __name__ == "__main__":
    qubit_ranges = range(2,3)
    gate_ranges = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99), (100, 119)]
    
    for num_qubit in qubit_ranges:
        for gate_range in tqdm(gate_ranges, desc="Gate Ranges", leave=False):
            filename = f"data/random_circuits_basis_rotations+cx_qubits_{num_qubit}_gates_{gate_range[0]}-{gate_range[1]}.pkl"

            with open(filename, 'rb') as file:
                data = pickle.load(file)

            # Calculate global expectation for each circuit
            result = []
            if isinstance(data[0], tuple):
                print("Data already labelled")
            else:

                num_qubits = data[0]['num_qubits']
                wire_labels = list(range(num_qubits))
                observable = generate_global_observable(len(wire_labels))
                for circuit in tqdm(data, desc="Processing Circuits"):
                    qasm_str = circuit['qasm']
                    pennylane_circuit = qml.from_qasm(qasm_str)
                    # Create a list of tuples
                    result.append((circuit, calculate_global_expectation(wire_labels, pennylane_circuit, observable)))

                # Save the updated data back to the original .pkl file
                with open(filename, 'wb') as file:
                    pickle.dump(result, file)

                print(f"Updated {len(result)} circuits with global expectation values and saved")
