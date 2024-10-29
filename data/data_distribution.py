import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

def load_entropy_data(directory, num_qubits):
    entropy_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl') and "qubits_"+str(num_qubits) in filename:
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
            if isinstance(data[0], tuple):
                entropy_data.extend([x[1] for x in data])
            else:
                continue
    return entropy_data

def plot_histograms(entropy_data, num_qubits):
    # Split the data into 5 clusters
    cluster_data = {i: entropy_data[i::5] for i in range(5)}
    averages = {key: round(np.mean(cluster), 3) for key, cluster in cluster_data.items()}

    # Plot the histogram for each cluster
    plt.figure(figsize=(10, 6))
    for key, cluster in cluster_data.items():
        counts, bin_edges = np.histogram(cluster, bins=20)
        normalized_counts = counts / counts.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.bar(bin_centers, normalized_counts, width=(bin_edges[1] - bin_edges[0]), alpha=1, label=f'Cluster {key}. Avg: {averages[key]}')
    
    plt.title(f'Stabilizer Rényi Entropy Distribution in {num_qubits}-qubit circuits ')
    plt.xlabel('Stabilizer Rényi Entropy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    filename = f"data/random_circuits/images/magic_distribution_q_{num_qubits}.png"
    plt.savefig(filename)
    print("Image saved as:", filename)

def plot_combined_histograms(directory, qubit_range):
    plt.figure(figsize=(10, 6))
    
    for num_qubits in qubit_range:
        entropy_data = load_entropy_data(directory, num_qubits)
        counts, bin_edges = np.histogram(entropy_data, bins=20)
        normalized_counts = counts / counts.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        plt.plot(bin_centers, normalized_counts, label=f'{num_qubits} qubits', alpha=0.7)
    
    plt.title('Stabilizer Rényi Entropy Distribution in the datasets')
    plt.xlabel('Stabilizer Rényi Entropy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    filename = "data/random_circuits/images/combined_magic_distribution.png"
    plt.savefig(filename)
    print("Combined image saved as:", filename)

if __name__ == "__main__":
    directory = 'data/random_circuits'
    for num_qubits in range(2, 6):
        entropy_data = load_entropy_data(directory, num_qubits)
        plot_histograms(entropy_data, num_qubits)
    
    # Plot combined histograms
    """qubit_range = range(2, 6)
    plot_combined_histograms(directory, qubit_range)
"""