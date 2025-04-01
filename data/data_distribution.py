import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

def load_entropy_data(directory, num_qubits):
    # Initialize as dictionary
    entropy_data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pkl') and "qubits_"+str(num_qubits) in filename:
            filepath = os.path.join(directory, filename)
            # Extract the number from filename
            if 'dataset_tim' in directory:
                key = int(filename.split('_trotter_')[1].split('.')[0])  # get number of trotter steps
            else:  # dataset_random
                # Extract the last two digits before .pkl
                key = int(filename.split('.pkl')[0][-2:])
                
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
            if isinstance(data[0], tuple):
                file_data = [x[1] for x in data]
                entropy_data[key] = file_data
            else:
                continue
    return entropy_data

def plot_histograms(entropy_data, num_qubits):
    # Calculate averages and standard deviations
    stats = {key: (round(np.mean(cluster), 3), round(np.std(cluster), 3)) 
            for key, cluster in entropy_data.items()}

    # Plot the histogram for each cluster
    plt.figure(figsize=(10, 6))
    # Sort the keys to plot in order
    sorted_keys = sorted(entropy_data.keys())
    
    # Find global min and max for consistent binning
    all_values = []
    for key in sorted_keys:
        all_values.extend(entropy_data[key])
    
    # Create common bin edges for all histograms
    n_bins = 20
    global_bin_edges = np.histogram(all_values, bins=n_bins)[1]
    global_bin_centers = (global_bin_edges[:-1] + global_bin_edges[1:]) / 2
    
    for key in sorted_keys:
        cluster = entropy_data[key]
        counts, _ = np.histogram(cluster, bins=global_bin_edges)
        normalized_counts = counts / counts.sum()

        plt.bar(global_bin_centers, normalized_counts, 
               width=(global_bin_edges[1] - global_bin_edges[0]), 
               alpha=0.9, label=f'Gates {key}. Avg: {stats[key][0]}, Std: {stats[key][1]}')
    
    # Set x-ticks to bin centers
    plt.xticks(global_bin_centers, rotation=45)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.title(f'Stabilizer Rényi Entropy Distribution in {num_qubits}-qubit circuits')
    plt.xlabel('Stabilizer Rényi Entropy')
    plt.ylabel('Frequency (%)')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    filename = directory+f"/images/magic_distribution_q_{num_qubits}.png"
    plt.savefig(filename)
    print("Image saved as:", filename)
    return stats, 

def plot_combined_histograms(directory, qubit_range):
    plt.figure(figsize=(10, 6))
    
    averages = []  # Store averages for later use
    for num_qubits in qubit_range:
        entropy_data = load_entropy_data(directory, num_qubits)
        entropy_data = [value for cluster in entropy_data.values() for value in cluster]
        average = np.mean(entropy_data)
        averages.append(average)
        counts, bin_edges = np.histogram(entropy_data, bins=20)
        normalized_counts = counts / counts.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot the distribution and get the color used
        line = plt.plot(bin_centers, normalized_counts, label=f'{num_qubits} qubits', alpha=0.7)[0]
        # Add vertical line with the same color
        plt.axvline(x=average, color=line.get_color(), linestyle='--', alpha=0.7)
        # Add average value annotation on the line
        plt.annotate(f'{average:.2f}', 
                    xy=(average, plt.ylim()[1]),
                    xytext=(average, plt.ylim()[1] * 1.05),
                    ha='center',
                    color=line.get_color())
    
    plt.title('Stabilizer Rényi Entropy Distribution in the dataset')
    plt.xlabel('Stabilizer Rényi Entropy')
    plt.ylabel('Frequency (%)')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set y-axis to percentage
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Create regular tick positions at 0.5 intervals for bottom x-axis
    regular_ticks = np.arange(0, max(bin_centers) + 0.2, 0.5)
    plt.xticks(regular_ticks, rotation=45)
    
    # Add some padding at the top for the annotations
    plt.margins(y=0.15)
    
    filename = directory+"/images/combined_magic_distribution.png"
    plt.savefig(filename)
    print("Combined image saved as:", filename)

if __name__ == "__main__":
    directories = ['dataset_tim', 'dataset_random']
    qubit_range = range(2, 7)
    for directory in directories:
        """for num_qubits in qubit_range:
            entropy_data = load_entropy_data(directory, num_qubits)
            plot_histograms(entropy_data, num_qubits)"""

        # Plot combined histograms
        plot_combined_histograms(directory, qubit_range)