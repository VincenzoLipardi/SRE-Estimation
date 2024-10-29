import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def save_results_as_image():
    # Directory containing the .pkl files
    directory = 'experiments'
    
    # List all .pkl files in the directory
    result_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    
    # Separate files into those with "pca" and those without
    pca_files = [f for f in result_files if 'pca' in f]
    non_pca_files = [f for f in result_files if 'pca' not in f and 'time' not in f]
    
    # Function to process files and create images
    def process_files(files, suffix):
        results_dict = {}
        estimators_dict = {}
        
        for result_file in files:
            with open(os.path.join(directory, result_file), 'rb') as file:
                result = pickle.load(file)
            
            qubit = int(result_file.split('_')[-1].replace('.pkl', ''))
            model_name = "Random_Forest"
            hyperparameters = result['hyperparameters']
            performance_metrics = result['performance_metrics']
            
            results_dict[qubit] = [(performance_metrics['MAE Test'], performance_metrics['MSE Test'], 
                                    performance_metrics['RMSE Test'], performance_metrics['R² Test'],
                                    performance_metrics['MAE Train'], performance_metrics['MSE Train'], 
                                    performance_metrics['RMSE Train'], performance_metrics['R² Train'])]
            estimators_dict[qubit] = [hyperparameters['n_estimators']]
        
        num_qubits = len(results_dict)
        fig, axes = plt.subplots(num_qubits, 1, figsize=(10, 4 * num_qubits))

        for ax, (qubit, results) in zip(axes, results_dict.items()):
            data = []
            for estimator, result in zip(estimators_dict[qubit], results):
                mae_test, mse_test, rmse_test, r2_test, mae_train, mse_train, rmse_train, r2_train = result
                data.append({
                    'Estimator': int(estimator),
                    'MAE Train': round(mae_train, 4),
                    'MSE Train': round(mse_train, 4),
                    'RMSE Train': round(rmse_train, 4),
                    'R² Train': round(r2_train, 4),
                    'MAE Test': round(mae_test, 4),
                    'MSE Test': round(mse_test, 4),
                    'RMSE Test': round(rmse_test, 4),
                    'R² Test': round(r2_test, 4)
                })
            
            df = pd.DataFrame(data)
            ax.axis('tight')
            ax.axis('off')
            ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            ax.set_title(f'Qubit: {qubit}')
        
        fig.suptitle(f'Model: {model_name}', fontsize=16)
        filename = f'experiments/images/results_random_{model_name}{suffix}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        qubits = sorted(results_dict.keys())
        mse_train = [results_dict[q][0][1] for q in qubits]
        mse_test = [results_dict[q][0][2] for q in qubits]

        plt.scatter(qubits, mse_train, color='blue', label='MSE Train')
        plt.scatter(qubits, mse_test, color='red', label='MSE Test')
        plt.plot(qubits, mse_train, color='blue', linestyle='-', linewidth=1)
        plt.plot(qubits, mse_test, color='red', linestyle='-', linewidth=1)
        
        plt.xlabel('Number of Qubits')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title(f'MSE for Model: {model_name}')
        plt.legend()
        plt.xticks(qubits)
        
        scatter_filename = f'experiments/images/mse_scatter_{model_name}{suffix}.png'
        plt.savefig(scatter_filename, bbox_inches='tight', dpi=300)
        plt.close()

    # Process PCA and non-PCA files separately
    process_files(pca_files, '_pca')
    process_files(non_pca_files, '')

save_results_as_image()
