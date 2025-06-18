import numpy as np
import pickle
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from typing import Tuple, Dict, Any
import os

class QuantumCircuitDAGDataset:
    """
    A class containing methods for converting quantum circuits to DAG representations.
    """
    
    def circuit_to_networkx_dag(self, qc: QuantumCircuit) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Convert a quantum circuit to a NetworkX DAG with node and edge features.
        
        Args:
            qc: Quantum circuit to convert
            
        Returns:
            Tuple of (NetworkX DAG, metadata dictionary)
        """
        dag = circuit_to_dag(qc)
        nx_dag = nx.DiGraph()
        
        # Node features mapping
        gate_type_mapping = {
            'rx': 0, 'ry': 1, 'rz': 2, 'cx': 3, 'h': 4, 'in': 5, 'out': 6, 'measure': 7
        }
        
        node_features = {}
        
        for node_id, node in enumerate(dag.topological_nodes()):
            # Node attributes
            if hasattr(node, 'op'):
                gate_name = node.op.name.lower()
                gate_type = gate_type_mapping.get(gate_name, 7)  
                
                # Get parameters if available
                params = []
                if hasattr(node.op, 'params') and node.op.params:
                    params = [float(p) for p in node.op.params]
                
                # Qubit information
                qubits = [qarg._index for qarg in node.qargs] if node.qargs else []
                
                node_features[node_id] = {
                    'gate_type': gate_type,
                    'gate_name': gate_name,
                    'params': params,
                    'num_params': len(params),
                    'qubits': qubits,
                    'num_qubits': len(qubits),
                    'is_parametric': len(params) > 0,
                    'is_two_qubit': len(qubits) == 2
                }
            else:
                # Input/Output nodes
                node_features[node_id] = {
                    'gate_type': gate_type_mapping.get('in', 5),  # Use 'in' type for input/output nodes
                    'gate_name': 'in_out',
                    'params': [],
                    'num_params': 0,
                    'qubits': [],
                    'num_qubits': 0,
                    'is_parametric': False,
                    'is_two_qubit': False
                }
            
            nx_dag.add_node(node_id, **node_features[node_id])
        
        # Add edges without features
        node_list = list(dag.topological_nodes())
        node_to_id = {node: i for i, node in enumerate(node_list)}
        
        for edge in dag.edges():
            source_id = node_to_id[edge[0]]
            target_id = node_to_id[edge[1]]
            nx_dag.add_edge(source_id, target_id)
        
        # Global features
        gate_counts = dict(qc.count_ops()) if qc.count_ops() else {}
        
        # Count each of the 8 gate types (rx, ry, rz, cx, h, in, out, measure)
        gate_type_counts = {
            'rx': gate_counts.get('rx', 0),
            'ry': gate_counts.get('ry', 0), 
            'rz': gate_counts.get('rz', 0),
            'cx': gate_counts.get('cx', 0),
            'h': gate_counts.get('h', 0),
            'measure': gate_counts.get('measure', 0)
        }
        
        # Metadata with global features
        metadata = {
            'num_qubits': qc.num_qubits,
            'total_gates': sum(gate_type_counts.values()),
            'gate_counts': gate_type_counts,
            'depth': qc.depth(),
            'global_features': [
                qc.num_qubits,                    # number of qubits
                sum(gate_type_counts.values()),   # total number of gates
                gate_type_counts['rx'],           # count of rx gates
                gate_type_counts['ry'],           # count of ry gates
                gate_type_counts['rz'],           # count of rz gates
                gate_type_counts['cx'],           # count of cx gates
                gate_type_counts['h'],            # count of h gates
                gate_type_counts['measure'],      # count of measure gates
                qc.depth()                        # circuit depth
            ]
        }
        
        return nx_dag, metadata
    



def load_circuits(file_path):
    """Load circuits from a pkl file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_updated_circuits(file_path, updated_circuits):
    """Save the updated circuits to a pkl file."""
    with open(file_path, 'wb') as f:
        pickle.dump(updated_circuits, f)


def qasm_to_dag_representation(qasm_str):
    """
    Convert QASM string to DAG representation suitable for GNNs.
    
    Args:
        qasm_str: QASM string representation of the circuit
        
    Returns:
        Dictionary containing DAG features
    """
    # Convert QASM to QuantumCircuit
    try:
        qc = QuantumCircuit.from_qasm_str(qasm_str)
    except AttributeError:
        # For newer versions of qiskit
        from qiskit.qasm2 import loads
        qc = loads(qasm_str)
    
    # Convert to DAG using our existing method
    dag, metadata = QuantumCircuitDAGDataset().circuit_to_networkx_dag(qc)
    
    # Extract node features for GNN
    node_features = []
    for node_id in dag.nodes():
        features = dag.nodes[node_id]
        # Create feature vector: [gate_type, num_params, num_qubits, is_parametric, is_two_qubit]
        feature_vector = [
            features['gate_type'],
            features['num_params'],
            features['num_qubits'],
            int(features['is_parametric']),
            int(features['is_two_qubit'])
        ]
        node_features.append(feature_vector)
    
    # Extract edge information
    edge_index = list(dag.edges())
    
    # Return DAG representation
    return {
        'dag_nodes': len(dag.nodes()),
        'dag_edges': len(dag.edges()),
        'node_features': node_features,
        'edge_index': edge_index,
        'global_features': metadata['global_features'],
        'gate_counts': metadata['gate_counts'],
        'circuit_depth': metadata['depth'],
        'num_qubits': metadata['num_qubits']
    }


def process_circuit_with_dag(circuit_data):
    """Process a single circuit and add DAG representation."""
    try:
        dag_features = qasm_to_dag_representation(circuit_data['qasm'])
        circuit_data['dag'] = dag_features
    except Exception as e:
        print(f"Warning: Could not process circuit DAG: {e}")
        # Set default empty DAG representation
        circuit_data['dag'] = {
            'dag_nodes': 0,
            'dag_edges': 0,
            'node_features': [],
            'edge_index': [],
            'global_features': [0] * 9,
            'gate_counts': {'rx': 0, 'ry': 0, 'rz': 0, 'cx': 0, 'h': 0, 'measure': 0},
            'circuit_depth': 0,
            'num_qubits': 0
        }


def process_circuits_file_with_dag(filename):
    """Process a single circuits file and add DAG representations."""
    print(f"Processing {filename} for DAG representation...")
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    
    circuits = load_circuits(filename)
    
    if not circuits or not isinstance(circuits[0], (dict, tuple)):
        print(f"Invalid circuit format in {filename}")
        return
    
    # Handle both (circuit_dict, label) tuples and direct dictionaries
    if isinstance(circuits[0], tuple):
        circuits_data = [circuit[0] for circuit in circuits]
    else:
        circuits_data = circuits
    
    # Check if DAG representation already exists
    has_dag = any('dag' in circuit for circuit in circuits_data)
    
    if has_dag:
        print(f"Skipping {filename}: DAG representation already present")
        return
    
    print(f"Adding DAG representation to {len(circuits_data)} circuits...")
    
    # Process each circuit
    for i, circuit_data in enumerate(circuits_data):
        process_circuit_with_dag(circuit_data)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(circuits_data)} circuits")
    
    # Save updated circuits
    save_updated_circuits(filename, circuits)
    print(f"✓ Updated {filename} with DAG representations")


def process_random_circuits_dag(qubit_ranges=range(2, 9), gate_ranges=[(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]):
    """Process random circuits dataset and add DAG representations."""
    print("=" * 60)
    print("Processing Random Circuits for DAG Representation")
    print("=" * 60)
    
    for num_qubit in qubit_ranges:
        print(f"\nProcessing {num_qubit}-qubit circuits...")
        for gate_range in gate_ranges:
            filename = f"data/dataset_random/basis_rotations+cx_qubits_{num_qubit}_gates_{gate_range[0]}-{gate_range[1]}.pkl"
            process_circuits_file_with_dag(filename)
    
    print("\nRandom circuits DAG processing completed!")


def process_ising_circuits_dag(min_qubit=2, max_qubit=7, min_trotter_step=1, max_trotter_step=11):
    """Process Ising model circuits dataset and add DAG representations."""
    print("=" * 60)
    print("Processing Ising Circuits for DAG Representation")
    print("=" * 60)
    
    for num_qubit in range(min_qubit, max_qubit):
        print(f"\nProcessing {num_qubit}-qubit Ising circuits...")
        for trotter_steps in range(min_trotter_step, max_trotter_step):
            filename = f"data/dataset_tim/ising_qubits_{num_qubit}_trotter_{trotter_steps}.pkl"
            process_circuits_file_with_dag(filename)
    
    print("\nIsing circuits DAG processing completed!")


def main():
    """
    Process existing circuit datasets and add DAG representations.
    """
    print("Quantum Circuit DAG Representation Processor")
    print("This script will add DAG representations to existing circuit datasets")
    print("following the same pattern as other feature representations.\n")
    
    # Ask user what to process
    print("What would you like to process?")
    print("1. Random circuit datasets")
    print("2. Ising (Tim) circuit datasets") 
    print("3. Both random and Ising circuit datasets")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        process_random_circuits_dag()
    elif choice == "2":
        process_ising_circuits_dag()
    elif choice == "3":
        process_random_circuits_dag()
        process_ising_circuits_dag()
    elif choice == "4":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Exiting...")
        return
    
    print("\n" + "=" * 60)
    print("DAG Representation Processing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Process both random and Ising circuits by default
    process_random_circuits_dag()
    process_ising_circuits_dag()

