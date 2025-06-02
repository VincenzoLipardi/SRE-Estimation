import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def load_results(model, min_qubits=2, max_qubits=6, results_subdir=''):
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
        file_path = f'experiments/{results_subdir}/results_{model}_qubit_{i}.pkl'
        
        
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
    


def plot_mse_histograms_grouped(model, results, feature_types, results_subdir=''):
    """
    Plot vertically aligned histograms of MSE values for each qubit, for datasets 'random' and 'tim'.
    For each qubit, show three groups (one for each feature type), each with two bars (train and test MSE).
    Color by feature type, hatch by train/test. Only includes results where dataset_limited == 'all'.
    """
    results_df = pd.DataFrame(results)
    datasets = ["random", "tim"]
    colors = ['#0072B2', '#E69F00', '#009E73']  # One color per feature type
    hatches = ['', '//']  # '' for train, '//' for test
    

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # More compact for two-column paper
    width = 0.12  # width of each bar
    n_features = len(feature_types)

    # Map feature_types to legend labels
    legend_labels = [ftype if ftype != 'gate_bins' else 'gate counts' for ftype in feature_types]

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        filtered_df = results_df[(results_df['dataset'] == dataset) & (results_df['dataset_limited'] == 'all')]
        filtered_df = filtered_df.sort_values(['qubits', 'features'])
        qubits = sorted(filtered_df['qubits'].unique())
        x = np.arange(len(qubits))

        for j, feature in enumerate(feature_types):
            mse_train = []
            mse_test = []
            for q in qubits:
                subdf = filtered_df[(filtered_df['qubits'] == q) & (filtered_df['features'] == feature)]
                mse_train.append(subdf['mse_train'].mean() if not subdf.empty else np.nan)
                mse_test.append(subdf['mse_test'].mean() if not subdf.empty else np.nan)
            # For each feature type, offset the group
            group_offset = (j - (n_features-1)/2) * (width*2.2)
            # Plot train
            ax.bar(x + group_offset - width/2, mse_train, width, label=f'{feature} Train' if i==0 else "", color=colors[j], hatch=hatches[0], edgecolor='black')
            # Plot test
            ax.bar(x + group_offset + width/2, mse_test, width, label=f'{feature} Test' if i==0 else "", color=colors[j], hatch=hatches[1], edgecolor='black')

        ax.set_ylabel('MSE')
        ax.set_title(f"Dataset: {dataset}")
        ax.set_xticks(x)
        ax.set_xticklabels(qubits)
        ax.grid(True, linestyle='--', axis='y', color='#cccccc', linewidth=0.7)
        # Set horizontal grid lines at 0.01 spacing
        ax.yaxis.set_major_locator(MultipleLocator(0.01))

    axes[1].set_xlabel('Number of Qubits')
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[k], edgecolor='black', label=legend_labels[k]) for k in range(n_features)
    ] + [
        Patch(facecolor='white', edgecolor='black', hatch=hatches[0], label='Train'),
        Patch(facecolor='white', edgecolor='black', hatch=hatches[1], label='Test')
    ]
    axes[0].legend(handles=legend_elements, loc='best', ncol=2 + n_features, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 1])
    out_path = f"experiments/{results_subdir}/images/{model}_mse_histograms.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_generalization_gates(models, datasets, base_dir="experiments/extrapolation", file_pattern="{model}_depth_generalization_{dataset}.pkl"):
    """
    Plot only the MSE (Train, Test, Extrapolation) metrics from multiple generalization study results
    for any dataset/model, all in the same grouped bar plot.
    Accepts lists of models and datasets, and builds the list of pickle paths internally.
    """
    # Build list of pickle paths from models and datasets
    pickle_paths = [
        os.path.join(base_dir, file_pattern.format(model=model, dataset=dataset))
        for model in models for dataset in datasets
    ]
    results = []
    for path in pickle_paths:
        df = pd.read_pickle(path)
        row = df.iloc[0]
        dataset = row['Dataset']
        model_type = row['Model Type']
        # Determine which columns to use
        if dataset == 'tim':
            test_metrics = row['performance test ']
            extrap_metrics = row['performance extrapolation ']
            # Robust train key search
            if 'performance train ' in row:
                train_metrics = row['performance train ']
            else:
                # Try to find a key containing 'train' and 'metric'
                train_keys = [c for c in row.index if 'train' in c.lower() and 'metric' in c.lower()]
                if train_keys:
                    train_metrics = row[train_keys[0]]
                else:
                    print(f"[ERROR] Could not find train metrics key in file: {path}")
                    print(f"Available keys: {list(row.index)}")
                    raise KeyError('No train metrics key found')
            mse_train = train_metrics[1]  # MSE is second in list
            mse_test = test_metrics[1]
            mse_extrap = extrap_metrics[1]
        elif dataset == 'random':
            # Robust train key search
            if 'Train Split Metrics (MAE, MSE, RMSE, R2)' in row:
                train_metrics = row['Train Split Metrics (MAE, MSE, RMSE, R2)']
            else:
                # Try to find a key containing 'train' and 'metric'
                train_keys = [c for c in row.index if 'train' in c.lower() and 'metric' in c.lower()]
                if train_keys:
                    train_metrics = row[train_keys[0]]
                else:
                    print(f"[ERROR] Could not find train metrics key in file: {path}")
                    print(f"Available keys: {list(row.index)}")
                    raise KeyError('No train metrics key found')
            test_metrics = row['Test Split Metrics (MAE, MSE, RMSE, R2)'] if 'Test Split Metrics (MAE, MSE, RMSE, R2)' in row else None
            if test_metrics is None:
                test_keys = [c for c in row.index if 'test' in c.lower() and 'metric' in c.lower()]
                if test_keys:
                    test_metrics = row[test_keys[0]]
                else:
                    print(f"[ERROR] Could not find test metrics key in file: {path}")
                    print(f"Available keys: {list(row.index)}")
                    raise KeyError('No test metrics key found')
            extrap_metrics = row['Unseen Metrics (MAE, MSE, RMSE, R2)'] if 'Unseen Metrics (MAE, MSE, RMSE, R2)' in row else None
            if extrap_metrics is None:
                extrap_keys = [c for c in row.index if ('extrap' in c.lower() or 'unseen' in c.lower()) and 'metric' in c.lower()]
                if extrap_keys:
                    extrap_metrics = row[extrap_keys[0]]
                else:
                    print(f"[ERROR] Could not find extrapolation metrics key in file: {path}")
                    print(f"Available keys: {list(row.index)}")
                    raise KeyError('No extrapolation metrics key found')
            mse_train = train_metrics[1]
            mse_test = test_metrics[1]
            mse_extrap = extrap_metrics[1]
        else:
            # Try to auto-detect columns
            test_col = [c for c in df.columns if 'test' in c.lower() and 'metric' in c.lower()][0]
            extrap_col = [c for c in df.columns if 'extrap' in c.lower() or 'unseen' in c.lower()][0]
            train_col = [c for c in df.columns if 'train' in c.lower() and 'metric' in c.lower()][0]
            test_metrics = row[test_col]
            extrap_metrics = row[extrap_col]
            train_metrics = row[train_col]
            mse_train = train_metrics[1]
            mse_test = test_metrics[1]
            mse_extrap = extrap_metrics[1]
        results.append((f"{model_type.upper()} {dataset}", mse_train, mse_test, mse_extrap))

    group_labels = [r[0] for r in results]
    mse_trains = [r[1] for r in results]
    mse_tests = [r[2] for r in results]
    mse_extraps = [r[3] for r in results]

    x = np.arange(len(results))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width, mse_trains, width, label='Train MSE', color='#0072B2')
    bars2 = ax.bar(x, mse_tests, width, label='Test MSE', color='#56B4E9')
    bars3 = ax.bar(x + width, mse_extraps, width, label='Extrapolation MSE', color='#E69F00')

    # Annotate bars
    for bars in [bars1, bars2, bars3]:
        for rect in bars:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Mean Squared Error')
    # ax.set_title('Generalization Study MSE (Train, Test, Extrapolation) for All Models/Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=20, ha='right')
    ax.legend()
    ax.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    os.makedirs('experiments/extrapolation/images', exist_ok=True)
    out_path = f'experiments/extrapolation/images/generalization_gates.png'
    plt.savefig(out_path)
    plt.close()
    print(f'Saved: {out_path}')

def plot_generalization_qubits(rfr_path, svr_path):
    """
    Plot grouped bars of MSE (Train, Test, Extrapolation) for both RFR and SVR models and both datasets.
    """
  

    # Helper to extract MSEs from a row
    def extract_mses(perf):
        return perf['MSE Train'], perf['MSE Test'], perf.get('MSE Extrapolation', perf.get('MSE External Test', None))

    # Read both files
    df_rfr = pd.read_pickle(rfr_path)
    df_svr = pd.read_pickle(svr_path)

    # Prepare data
    labels = []
    mse_trains = []
    mse_tests = []
    mse_extrs = []
    for model_name, df in zip(['RFR', 'SVR'], [df_rfr, df_svr]):
        for i, row in df.iterrows():
            dataset = row['Dataset']
            perf = row['performance_metrics']
            mse_train, mse_test, mse_extr = extract_mses(perf)
            labels.append(f"{model_name} {dataset}")
            mse_trains.append(mse_train)
            mse_tests.append(mse_test)
            mse_extrs.append(mse_extr)

    x = np.arange(len(labels))
    width = 0.22

    os.makedirs('experiments/extrapolation/images', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width, mse_trains, width, label='Train MSE', color='#0072B2')
    bars2 = ax.bar(x, mse_tests, width, label='Test MSE', color='#56B4E9')
    bars3 = ax.bar(x + width, mse_extrs, width, label='Extrapolation MSE', color='#E69F00')

    # Annotate bars
    for bars in [bars1, bars2, bars3]:
        for rect in bars:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Mean Squared Error')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.legend()
    ax.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.tight_layout()
    out_path = 'experiments/extrapolation/images/generalization_qubits_compare.png'
    plt.savefig(out_path)
    plt.close()
    print(f'Saved: {out_path}')

def main(model, feature_set, min_qubits, max_qubits, results_subdir):

    
    for results_subdir in ['0.1', '0.2', '0.3', '0.4', '0.5']:
        results = load_results(model, min_qubits, max_qubits, results_subdir=results_subdir)
        plot_mse_histograms_grouped(model, results, feature_set, results_subdir=results_subdir)
    #plot_generalization_gates(models=["rfr", "svr"], datasets=["random", "tim"])
    """plot_generalization_qubits(
        "experiments/extrapolation/results_rfr_train_qubits_2-3-4-5_test_qubit_6.pkl",
        "experiments/extrapolation/results_svr_train_qubits_2-3-4-5_test_qubit_6.pkl")"""
    #plot_generalization_gates(models=["rfr", "svr"], datasets=["random", "tim"])

if __name__ == "__main__":
    main(model="svr", feature_set=["gate_bins", "shadow", "combined"], min_qubits=2, max_qubits=6, results_subdir='paper')

