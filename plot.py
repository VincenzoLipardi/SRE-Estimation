import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(min_qubits=2, max_qubits=6):
    """
    Load R-squared results from saved model files for various experiments.
    
    Parameters:
    - min_qubits: Minimum number of qubits to consider
    - max_qubits: Maximum number of qubits to consider
    
    Returns:
    - Dictionary containing R-squared results organized by dataset, feature type, and qubit count
    """
    results = {
        'dataset': [],
        'features': [],
        'qubits': [],
        'r2_train': [],
        'r2_test': [],
        'mse_train': [],
        'mse_test': [],
        'dataset_limited': []
    }
    
    for i in range(min_qubits, max_qubits + 1):
        file_path = f'experiments/results_qubit_{i}.pkl'
        
        if os.path.exists(file_path):
            try:
                df = pd.read_pickle(file_path)
                
                # Extract results from the DataFrame
                for _, row in df.iterrows():
                    results['dataset'].append(row['Dataset'])
                    results['features'].append(row['features'])
                    results['qubits'].append(i)
                    results['dataset_limited'].append(row['Dataset_limited'])
                    
                    # Extract R-squared and MSE scores
                    metrics = row['performance_metrics']
                    results['r2_train'].append(metrics['R² Train'])
                    results['r2_test'].append(metrics['R² Test'])
                    results['mse_train'].append(metrics['MSE Train'])
                    results['mse_test'].append(metrics['MSE Test'])
                    
                print(f"Loaded results from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    return results

def plot_r2_scores(results):
    """
    Create plots to visualize R-squared and MSE scores across different configurations.
    Only plots results where dataset_limited="all"
    
    Parameters:
    - results: Dictionary containing the extracted R-squared results
    """
    results_df = pd.DataFrame(results)
    
    # Create a figure with 2x2 subplots (2 rows for datasets × 2 columns for feature types)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('R-squared and MSE Scores for Full Datasets', fontsize=16)
    
    datasets = ["random", "tim"]
    feature_types = ["shadow", "gate_counts"]
    
    for i, dataset in enumerate(datasets):
        for j, feature_type in enumerate(feature_types):
            ax1 = axes[i, j]
            
            # Create second y-axis sharing the same x-axis
            ax2 = ax1.twinx()
            
            # Filter data
            filtered_df = results_df[(results_df['dataset'] == dataset) & 
                                   (results_df['features'] == feature_type) &
                                   (results_df['dataset_limited'] == 'all')]
            
            filtered_df = filtered_df.sort_values('qubits')
            
            # Plot R² on the left y-axis (ax1)
            line1, = ax1.plot(filtered_df['qubits'], filtered_df['r2_train'], 'o-', 
                            label='Train R²', color='blue')
            line2, = ax1.plot(filtered_df['qubits'], filtered_df['r2_test'], 's-', 
                            label='Test R²', color='purple')
            
            # Plot MSE on the right y-axis (ax2)
            line3, = ax2.plot(filtered_df['qubits'], filtered_df['mse_train'], 'o--', 
                            label='Train MSE', color='red')
            line4, = ax2.plot(filtered_df['qubits'], filtered_df['mse_test'], 's--', 
                            label='Test MSE', color='orange')
            
            # Set labels and title
            ax1.set_xlabel('Number of Qubits')
            ax1.set_ylabel('R-squared Score')
            ax2.set_ylabel('Mean Squared Error')
            
            ax1.set_title(f'Dataset: {dataset}, Features: {feature_type}')
            
            # Set x-ticks
            ax1.set_xticks(filtered_df['qubits'].unique())
            
            # Add grid (only for the left axis to avoid cluttering)
            ax1.grid(True, linestyle='--',)
            
            # Combine legends from both axes
            lines = [line1, line2, line3, line4]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')
            
            # Color the y-axis labels to match the plots
            ax1.yaxis.label.set_color('darkblue')
            ax2.yaxis.label.set_color('red')
            
            # Color the tick labels
            ax1.tick_params(axis='y', colors='darkblue')
            ax2.tick_params(axis='y', colors='red')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('experiments/images/full_dataset.png')
    plt.show()
    
    # Also create a combined plot showing dataset_limited influence
    plot_dataset_limited_influence(results_df)

def plot_dataset_limited_influence(results_df):
    """
    Plot the influence of dataset_limited parameter on R-squared scores.
    
    Parameters:
    - results_df: DataFrame containing all results
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Influence of Dataset Limitation on R-squared Test Scores', fontsize=16)
    
    datasets = ["random", "tim"]
    feature_types = ["shadow", "gate_counts"]
    
    for i, dataset in enumerate(datasets):
        for j, feature_type in enumerate(feature_types):
            ax = axes[i, j]
            
            # Filter data for current dataset and feature type
            filtered_df = results_df[(results_df['dataset'] == dataset) & 
                                     (results_df['features'] == feature_type)]
            
            # Get unique dataset_limited values and qubits
            # Custom sort to handle mix of 'all' string and integer values
            limit_values = filtered_df['dataset_limited'].unique()
            # Sort limit values by putting 'all' first, then numerical values
            limit_values = sorted([x for x in limit_values if x == 'all']) + sorted([x for x in limit_values if x != 'all'], key=lambda x: (isinstance(x, str), x))
            
            qubit_values = sorted(filtered_df['qubits'].unique())
            
            # Plot lines for each dataset_limited value
            for limit in limit_values:
                limit_data = filtered_df[filtered_df['dataset_limited'] == limit]
                if not limit_data.empty:
                    # Group by qubits
                    grouped = limit_data.groupby('qubits')['r2_test'].mean().reset_index()
                    grouped = grouped.sort_values('qubits')
                    
                    label = 'All data' if limit == 'all' else f'Limit: {limit}'
                    ax.plot(grouped['qubits'], grouped['r2_test'], 'o-', label=label)
            
            ax.set_title(f'Dataset: {dataset}, Features: {feature_type}')
            ax.set_xlabel('Number of Qubits')
            ax.set_ylabel('Test R-squared Score')
            ax.set_xticks(qubit_values)
            ax.grid(True, linestyle='--')
            ax.legend(title='Dataset Limitation')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('experiments/images/dataset_limited_influence.png')
    plt.show()

def main(min_qubits=2, max_qubits=6):
    # Create output directory if it doesn't exist
    os.makedirs('experiments/images', exist_ok=True)
    
    # Load results from pickle files
    results = load_results(min_qubits, max_qubits)
    
    """# Display summary statistics
    results_df = pd.DataFrame(results)
    print("\nSummary of R-squared results:")
    print(results_df.groupby(['dataset', 'features', 'qubits']).agg({
       'r2_train': ['mean', 'std', 'count'],
        'r2_test': ['mean', 'std', 'count']
    }))"""
    
    # Plot results
    plot_r2_scores(results)
    
    # Save the results DataFrame to a CSV file for further analysis
    #results_df.to_csv('experiments/r2_results_summary.csv', index=False)
    #print("\nResults saved to 'experiments/r2_results_summary.csv'")

if __name__ == "__main__":
    main()
