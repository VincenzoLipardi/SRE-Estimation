import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# Function to load data from .pkl files
def load_data(directory, num_qubit):
    labels, data = [], []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl') and "qubits_"+str(num_qubit) in filename:
            # print(filename)
            with open(os.path.join(directory, filename), 'rb') as file:
                circuit_data = pickle.load(file)
                # print(circuit_data[0][0].keys())
                # Assume circuit_data is a tuple (list_of_dicts, label, label)
                for circuit in circuit_data:
                    labels.append(circuit[2])
                    data.append([float(v) for k, v in circuit[0].items() if 'obs_' in k and v is not None])
    
    return np.array(data), labels

# Function to train Random Forest on data
def train_random_forest(data, labels, estimators):
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
    rfr = RandomForestRegressor(n_estimators=estimators, 
                                min_samples_leaf=1, 
                                min_samples_split=2, 
                                random_state=42)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)

    # Calculate metrics for test set
    mean_absolute_error_test = mean_absolute_error(y_test, y_pred)
    mean_squared_error_test = mean_squared_error(y_test, y_pred)
    root_mean_squared_error_test = root_mean_squared_error(y_test, y_pred)
    r_squared_test = r2_score(y_test, y_pred)

    # Calculate metrics for training set
    y_train_predictions = rfr.predict(X_train)
    mean_absolute_error_train = mean_absolute_error(y_train, y_train_predictions)
    mean_squared_error_train = mean_squared_error(y_train, y_train_predictions)
    root_mean_squared_error_train = root_mean_squared_error(y_train, y_train_predictions)
    r_squared_train = r2_score(y_train, y_train_predictions)

    print("\nTraining Set:")
    print(f"Mean Absolute Error: {mean_absolute_error_train:.4f}")
    print(f"Mean Squared Error: {mean_squared_error_train:.4f}")
    print(f"Root Mean Squared Error: {root_mean_squared_error_train:.4f}")
    print(f"R-squared Score: {r_squared_train:.4f}")

    print("Test Set:")
    print(f"Mean Absolute Error: {mean_absolute_error_test:.4f}")
    print(f"Mean Squared Error: {mean_squared_error_test:.4f}")
    print(f"Root Mean Squared Error: {root_mean_squared_error_test:.4f}")
    print(f"R-squared Score: {r_squared_test:.4f}")
    return (mean_absolute_error_test, mean_squared_error_test, root_mean_squared_error_test, r_squared_test,
            mean_absolute_error_train, mean_squared_error_train, root_mean_squared_error_train, r_squared_train)

def save_results_as_image(results_dict, estimators_dict, model_name):
    # Create a figure with subplots for each qubit
    num_qubits = len(results_dict)
    fig, axes = plt.subplots(num_qubits, 1, figsize=(10, 4 * num_qubits))
    
    if num_qubits == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    for ax, (qubit, results) in zip(axes, results_dict.items()):
        # Unpack the results
        mae_test, mse_test, rmse_test, r2_test, mae_train, mse_train, rmse_train, r2_train = results
        
        # Create a DataFrame
        df = pd.DataFrame({
            'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
            'Training': [round(mae_train, 4), round(mse_train, 4), round(rmse_train, 4), round(r2_train, 4)],
            'Test': [round(mae_test, 4), round(mse_test, 4), round(rmse_test, 4), round(r2_test, 4)]
        })
        
        # Plot the DataFrame as a table
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        ax.set_title(f'Qubit: {qubit}, Estimators: {estimators_dict[qubit]}')
    
    # Add a main title to the figure
    fig.suptitle(f'Model: {model_name}', fontsize=16)
    
    # Save the table as a PNG image
    filename='results_'+model_name+'.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Main function to execute the process
def main():
    directory = 'data'
    results_dict = {}
    estimators_dict = {}
    model_name = 'Random_Forest'  # Specify the model name here
    for num_qubit in range(2, 4):  # Loop over qubits 2 to 3
        data, labels = load_data(directory, num_qubit)
        print(f"Data size for qubit {num_qubit}:", len(data), "Label size:", len(labels))
        if num_qubit == 2:
            num_estimators = 15
        elif num_qubit == 3:
           num_estimators = 64
        elif num_qubit == 4:
           num_estimators = 512
        results = train_random_forest(data, labels, estimators=num_estimators)
        results_dict[num_qubit] = results
        estimators_dict[num_qubit] = num_estimators
    
    save_results_as_image(results_dict, estimators_dict, model_name)

if __name__ == "__main__":
    main()
