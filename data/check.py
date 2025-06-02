import os
import pickle
from tabulate import tabulate
import matplotlib.pyplot as plt
import glob

def get_feature_length(file_path):
    """Load the pickle file and return the feature length or an error message."""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if data:
                return str(len(data[0][0]))
            else:
                return "empty data"
    else:
        return "file not found"

def get_feature_type(file_path):
    """Load the pickle file and return the type of data[0][0] or an error message."""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if data:
                return str(type(data[0][0]))
            else:
                return "empty data"
    else:
        return "file not found"

def collect_data(qubit_nums, subdirs, parent_dir):
    """Collect feature lengths for each qubit and subdir."""
    rows = []
    for q in qubit_nums:
        filename = f"basis_rotations+cx_qubits_{str(q)}_gates_20-39.pkl"
        row = [q]
        for subdir in subdirs:
            file_path = os.path.join(parent_dir, subdir, filename)
            row.append(get_feature_length(file_path))
        rows.append(row)
    return rows

def print_summary(rows, subdirs):
    """Print one summary line per qubit."""
    for row in rows:
        q = row[0]
        print(f"Qubits={q}: " + ", ".join([f"{subdir}: {row[i+1]}" for i, subdir in enumerate(subdirs)]))

def print_feature_types(qubit_nums, subdirs, parent_dir):
    """Print the type of data[0][0] for each qubit and subdir."""
    print("\nFeature type for data[0][0] in each subdirectory:")
    for q in qubit_nums:
        filename = f"basis_rotations+cx_qubits_{str(q)}_gates_20-39.pkl"
        types = []
        for subdir in subdirs:
            file_path = os.path.join(parent_dir, subdir, filename)
            types.append(get_feature_type(file_path))
        print(f"Qubits={q}: " + ", ".join([f"{subdir}: {types[i]}" for i, subdir in enumerate(subdirs)]))

def print_table(rows, header):
    """Print the table in the terminal."""
    print("\nSummary Table:")
    print(tabulate(rows, headers=header, tablefmt="grid"))

def save_table_image(rows, header, image_path, title):
    """Save the table as an image with a title."""
    fig, ax = plt.subplots(figsize=(len(header), len(rows)*0.5))
    ax.axis('off')
    plt.title(title, fontsize=10, pad=20)
    table = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(header))))
    # Set row height
    row_height = 0.15
    for key, cell in table.get_celld().items():
        cell.set_height(row_height)
    plt.tight_layout()
    plt.savefig(image_path)
    print(f"Table image saved as {image_path}")

def check_qubit2_feature_dimensions():
    """Check if all files in 'combined' for qubit_num=2 have features of the same dimension and print the result."""
    combined_dir = os.path.join("data", "dataset_random", "combined")
    pattern = os.path.join(combined_dir, "basis_rotations+cx_qubits_2_gates_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No files found for qubit_num=2 in 'combined' directory.")
        return
    
    feature_dims = {}
    for file_path in files:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if data and data[0] and hasattr(data[0], '__getitem__'):
                    dim = len(data[0][0])
                else:
                    dim = 'empty or invalid data'
        except Exception as e:
            dim = f'error: {e}'
        feature_dims[file_path] = dim
    
    dims = set(feature_dims.values())
    if len(dims) == 1:
        print(f"All files for qubit_num=2 in 'combined' have the same feature dimension: {dims.pop()}")
    else:
        print("Files for qubit_num=2 in 'combined' have differing feature dimensions:")
        for fname, dim in feature_dims.items():
            print(f"  {os.path.basename(fname)}: {dim}")

def print_qubit2_gates_0_19_content():
    """Open and print the content of data/dataset_random/basis_rotations+cx_qubits_3_gates_0-19.pkl."""
    file_path = os.path.join("data", "dataset_random/", "basis_rotations+cx_qubits_2_gates_0-19.pkl")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Content of {file_path}:")
            print(data[10])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def main():
    parent_dir = "data/dataset_random"
    subdirs = ["gate_bins", "shadow", "combined"]
    qubit_num = [2, 3, 4, 5, 6]
    header = ["Qubits"] + subdirs

    # check_qubit2_feature_dimensions()
    print_qubit2_gates_0_19_content()

if __name__ == "__main__":
    main()
