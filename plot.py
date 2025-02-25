import os
import pandas as pd
import matplotlib.pyplot as plt

def save_results_as_image(model, data, pca, gates):
    # Directory containing the .pkl files
    directory = 'experiments'
    
    # List all .pkl files in the directory
    result_files = [f for f in os.listdir(directory) if f.endswith('.pkl') and data in f and "time" not in f]
    # print(result_files)
    # Function to process files and create images
    def read_file(file, pca, gates):
        with open(os.path.join(directory, file), 'rb') as file_obj:
            df = pd.read_pickle(file_obj)
        print(df.shape)
        # print(df.columns)

        qubit = int(file.split('_')[-1].replace('.pkl', ''))
        filtered_df = df[(df['pca'] == pca) & (df['Dataset'].str.contains(str(gates)))]
        print("Filtered DataFrame shape:", filtered_df.shape)
        print("Filtered DataFrame contents:\n", filtered_df)
        if filtered_df.shape[0] != 1:
            raise ValueError("More than one model has been chosen to be plotted")
        metrics = filtered_df["performance_metrics"]
        return qubit, metrics

    results_for_qubits = {}
    for file in result_files:
        qubit, metrics = read_file(file, pca, gates)
        results_for_qubits[qubit] = metrics


    plt.figure(figsize=(10, 6))
    qubits = sorted(results_for_qubits.keys())
    mse_train = [results_for_qubits[q]["MSE Train"] for q in qubits]
    mse_test = [results_for_qubits[q]["MSE Test"] for q in qubits]

    plt.scatter(qubits, mse_train, color='blue', label='MSE Train')
    plt.scatter(qubits, mse_test, color='red', label='MSE Test')
    plt.plot(qubits, mse_train, color='blue', linestyle='-', linewidth=1)
    plt.plot(qubits, mse_test, color='red', linestyle='-', linewidth=1)
    
    plt.xlabel('Number of Qubits')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'MSE for Model: {model}')
    plt.legend()
    plt.xticks(qubits)
    
    scatter_filename = f'experiments/images/mse_scatter_{data}_{model}_{gates}.png'
    plt.savefig(scatter_filename, bbox_inches='tight', dpi=300)
    plt.close()

pca = False
model = "Random_Forest"
data = "tim"
gates = 1

save_results_as_image(model, data, pca, gates)
