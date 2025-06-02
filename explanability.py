import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import os
from lime.lime_tabular import LimeTabularExplainer



def run_shap_explanation(model_path, data_dir, image_dir, num_qubit=3, sample_size=10):
    # Load the trained model (pick the last model in the DataFrame)
    df = pd.read_pickle(model_path)
    model = df.iloc[0]['Model']

    # Load features and labels for qubit 3 (use all gates)
    def load_gate_bins_features(data_dir, num_qubit=3):
        features, labels = [], []
        pattern = f"qubits_{num_qubit}"
        for filename in os.listdir(data_dir):
            if filename.endswith('.pkl') and pattern in filename:
                with open(data_dir / filename, 'rb') as f:
                    data = pickle.load(f)
                    for feature, label in data:
                        if isinstance(feature, dict):
                            feature = list(feature.values())
                        features.append(feature)
                        labels.append(label)
        X = np.array(features)
        y = np.array(labels)
        return X, y

    X, y = load_gate_bins_features(data_dir, num_qubit=num_qubit)

    # Use a subset for SHAP (for speed)
    X_sample = X[:sample_size]

    # Compute SHAP values
    explainer = shap.TreeExplainer(model, feature_perturbation='auto')
    shap_values = explainer(X_sample, check_additivity=False)

    # Plot and save global feature importance bar chart
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(image_dir / "shap_summary_bar.png")
    plt.close()

    # Plot and save global SHAP summary (beeswarm) plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(image_dir / "shap_summary_beeswarm.png")
    plt.close()

    # Plot and save force plot for the first sample
    shap_value_single = shap_values[0]
    expected_value = explainer.expected_value

    # Use matplotlib-based force plot (non-interactive)
    plt.figure()
    shap.force_plot(expected_value, shap_value_single, X_sample[0], matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(image_dir / "shap_force_sample_0.png")
    plt.close()

    # Plot and save SHAP waterfall plot for the first sample
    plt.figure()
    shap.plots.waterfall(shap_values, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(image_dir / "shap_waterfall_sample_0.png")
    plt.close()





def run_lime_explanation(model_path, data_dir, image_dir, num_qubit=3, sample_idx=0, num_features=10):
    """
    Load model and data, then run LIME local explanation for a single sample and save the plot.
    """
    # Load the trained model (pick the last model in the DataFrame)
    df = pd.read_pickle(model_path)
    model = df.iloc[0]['Model']

    # Load features and labels for the specified qubit
    def load_gate_bins_features(data_dir, num_qubit=3):
        features, labels = [], []
        pattern = f"qubits_{num_qubit}"
        for filename in os.listdir(data_dir):
            if filename.endswith('.pkl') and pattern in filename:
                with open(data_dir / filename, 'rb') as f:
                    data = pickle.load(f)
                    for feature, label in data:
                        if isinstance(feature, dict):
                            feature = list(feature.values())
                        features.append(feature)
                        labels.append(label)
        X = np.array(features)
        y = np.array(labels)
        return X, y

    X, y = load_gate_bins_features(data_dir, num_qubit=num_qubit)

    # Create the LIME explainer
    explainer = LimeTabularExplainer(
        X,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
        discretize_continuous=True,
        mode='regression'  
    )

    # Explain the prediction for the selected sample
    exp = explainer.explain_instance(
        X[sample_idx],
        model.predict,  
        num_features=num_features
    )
    # Print the model's prediction for sample 0
    print(f"Prediction for sample {sample_idx}: {model.predict([X[sample_idx]])}")
    print(f"Actual label for sample {sample_idx}: {y[sample_idx]}")
    print(f"Features for sample {sample_idx}: {X[sample_idx]}")
    print(X[sample_idx][96], X[sample_idx][83], X[sample_idx][68], X[sample_idx][21], X[sample_idx][57], X[sample_idx][34])
    # Save the explanation as an image
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(image_dir / f"lime_local_sample_{sample_idx}.png")
    plt.close()





if __name__ == "__main__":
    model_path = Path('experiments/results_rfr_qubit_3.pkl')
    data_dir = Path('data/dataset_random/gate_bins')
    image_dir = Path("experiments/images/explanation")
    
    run_lime_explanation(model_path, data_dir, image_dir, num_qubit=3, sample_idx=8000, num_features=5)
