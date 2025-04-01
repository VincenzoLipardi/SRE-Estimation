import os
import pickle
import numpy as np

def load_circuits(file_path):
    """Load circuits from a pkl file."""
    with open(file_path, 'rb') as f:
        circuits = pickle.load(f)
    return circuits

def save_circuits(file_path, circuits):
    """Save circuits to a pkl file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(circuits, f)

def extract_features(circuit_tuple):
    """Extract different feature sets from a circuit dictionary."""
    # Extract shadows (observable expectation values)
    
    circuit_dict, sre = circuit_tuple
    shadows = [float(v) for k, v in circuit_dict.items() if 'obs_' in k and v is not None]
    
    # Extract DAG
    dag = circuit_dict.get('dag', None)
    
    # Extract gate counts
    feature_to_extract = ["h", "cx", "rx", "ry", "rz"]
    num_qubits = circuit_dict.get('num_qubits', None)
    gate_counts = circuit_dict.get('gate_counts', {})
    for i in range(num_qubits):
        feature_to_extract.append(f"rx_q{i}")
        feature_to_extract.append(f"ry_q{i}")
        feature_to_extract.append(f"rz_q{i}")
    
    # Use get() method with default value 0 for missing keys
    gate_counts = [gate_counts.get(feature, 0) for feature in feature_to_extract]
    
    return shadows, dag, gate_counts, sre

def check_and_create_subfolders(dataset_folder):
    """Check if feature subfolders exist and create them if they don't."""
    feature_folders = ['shadow', 'dag', 'gate_counts']
    
    for feature in feature_folders:
        feature_path = os.path.join(dataset_folder, feature)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
            print(f"Created new feature folder: {feature_path}")
        else:
            print(f"Feature folder already exists: {feature_path}")

def process_dataset(dataset_folder):
    """Process all pkl files in a dataset folder."""
    # First check and create necessary subfolders
    check_and_create_subfolders(dataset_folder)
    
    # List all pkl files in the folder
    pkl_files = [f for f in os.listdir(dataset_folder) if f.endswith('.pkl') and os.path.isfile(os.path.join(dataset_folder, f))]
    
    for file in pkl_files:
        file_path = os.path.join(dataset_folder, file)
        # print(f"Processing {file_path}")
        
        # Load the original circuits
        circuits = load_circuits(file_path)
        
        # Initialize lists for different feature sets
        shadow_dataset = []
        dag_dataset = []
        gate_counts_dataset = []
        
        # Process each circuit
        for circuit_tuple in circuits:
            if isinstance(circuit_tuple, tuple):
                shadows, dag, gate_counts, sre = extract_features(circuit_tuple)
                
                # Create tuples for each feature set with the corresponding label
                if shadows:
                    shadow_dataset.append((shadows, sre))
                if dag:
                    dag_dataset.append((dag, sre))
                if gate_counts:
                    gate_counts_dataset.append((gate_counts, sre))
        
        # Generate new filenames in their respective subfolders
        base_name = os.path.basename(file)
        shadow_file = os.path.join(dataset_folder, 'shadow', base_name)
        dag_file = os.path.join(dataset_folder, 'dag', base_name)
        gate_counts_file = os.path.join(dataset_folder, 'gate_counts', base_name)
        
        # Save the separate feature datasets
        if shadow_dataset:
            save_circuits(shadow_file, shadow_dataset)
            print(f"Saved shadows dataset to {shadow_file}")
        
        if dag_dataset:
            save_circuits(dag_file, dag_dataset)
            print(f"Saved DAG dataset to {dag_file}")
        
        if gate_counts_dataset:
            save_circuits(gate_counts_file, gate_counts_dataset)
            print(f"Saved gate counts dataset to {gate_counts_file}")

def main():
    # Process dataset folders
    datasets = ['dataset_tim']
    
    for dataset in datasets:
        if not os.path.exists(dataset):
            os.makedirs(dataset)
            print(f"Created new dataset folder: {dataset}")
        
        print(f"\nProcessing {dataset}")
        process_dataset(dataset)

if __name__ == "__main__":
    main()