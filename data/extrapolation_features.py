import os
import pickle
import glob
import re

# Directories to process
folders = [
    ("dataset_random/gate_bins", "dataset_random/gate_bins_ext"),
    ("dataset_tim/gate_bins", "dataset_tim/gate_bins_ext")
]

def extract_qubits_from_filename(filename):
    # Looks for 'qubits_X' in the filename
    match = re.search(r"qubits_(\d+)", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract qubits from filename: {filename}")

def one_hot_qubits(n_qubits):
    # For qubits 2-6, returns a 5-dim one-hot vector
    if n_qubits < 2 or n_qubits > 6:
        raise ValueError(f"Number of qubits {n_qubits} out of range (2-6)")
    vec = [0]*5
    vec[n_qubits-2] = 1
    return vec

def process_file(input_path, output_path, n_qubits):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    new_data = []
    qubit_vec = one_hot_qubits(n_qubits)
    for row in data:
        features, value = row
        new_features = features + qubit_vec
        new_data.append((new_features, value))
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)

def main():
    for src_dir, dst_dir in folders:
        os.makedirs(dst_dir, exist_ok=True)
        for file in glob.glob(os.path.join(src_dir, '*.pkl')):
            filename = os.path.basename(file)
            try:
                n_qubits = extract_qubits_from_filename(filename)
                if n_qubits < 2 or n_qubits > 6:
                    print(f"Skipping {file}: number of qubits {n_qubits} out of range (2-6)")
                    continue
            except ValueError as e:
                print(e)
                continue
            out_path = os.path.join(dst_dir, filename)
            print(f"Processing {file} (qubits={n_qubits}) -> {out_path}")
            process_file(file, out_path, n_qubits)

if __name__ == "__main__":
    main()
