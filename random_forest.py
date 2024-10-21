import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# Function to load data from .pkl files
def load_data(directory, num_qubit):
    labels, data = [], []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl') and "qubits_"+str(num_qubit) in filename:
            print(filename)
            with open(os.path.join(directory, filename), 'rb') as file:
                circuit_data = pickle.load(file)
                # Assume circuit_data is a tuple (list_of_dicts, label)
                for circuit in circuit_data:
                    labels.append(circuit[1])
                    data.append([v for k, v in circuit[0].items() if k.startswith('obs_')])
    
    return data, labels

# Function to train Random Forest on data
def train_random_forest(data, labels):
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
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
    
    
    return y_pred, mean_absolute_error_test, mean_squared_error_test, root_mean_squared_error_test, r_squared_test

# Main function to execute the process
def main():
    directory = 'data'
    data, labels = load_data(directory, num_qubit=2)
    # print("Data size:", len(data), "Label size:", len(labels))
    train_random_forest(data, labels)

if __name__ == "__main__":
    main()
