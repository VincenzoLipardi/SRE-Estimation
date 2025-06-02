import os
import pickle
import glob

# Directories to process
folders = [
    ("data/dataset_random/gate_bins", "data/dataset_random/gate_bins_10"),
    #("data/dataset_tim/gate_bins", "data/dataset_tim/gate_bins_10")
]

def transform_features(features):
    # features: list of 152 floats
    # Keep first 2 unchanged

    first_two = features[:2]
    # Next 150: sum 5 by 5
    summed = [sum(features[2+i*5:2+(i+1)*5]) for i in range(30)]
    return first_two + summed

def process_file(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    # Assume data is a list of tuples: (features, float)
    new_data = []
    for row in data:
        # row: (features, float)
        features, value = row
        new_features = transform_features(features)
        new_data.append((new_features, value))
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)

def main():
    for src_dir, dst_dir in folders:
        os.makedirs(dst_dir, exist_ok=True)
        for file in glob.glob(os.path.join(src_dir, '*.pkl')):
            filename = os.path.basename(file)
            out_path = os.path.join(dst_dir, filename)
            print(f"Processing {file} -> {out_path}")
            process_file(file, out_path)

if __name__ == "__main__":
    main()
