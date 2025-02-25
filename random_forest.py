import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.decomposition import PCA

# Function to load data from .pkl files
def load_data(directory, num_qubit, dataset_limited="all"):
    labels, data = [], []
    filename_to_save = None  # Initialize with a default value
    for filename in os.listdir(directory):
        if dataset_limited is not False:
            if isinstance(dataset_limited, str):
                condition = filename.endswith('.pkl') and "qubits_" + str(num_qubit) in filename and dataset_limited in filename
            elif isinstance(dataset_limited, int):
                condition = filename.endswith('.pkl') and "qubits_" + str(num_qubit) in filename and str(dataset_limited) in filename
            else:
                raise TypeError("dataset_limited has to be or str or int type. Or, pick False if you want to study the entire dataset")
        else:
            condition = filename.endswith('.pkl') and "qubits_" + str(num_qubit) in filename
        if condition:
            filename_to_save = filename
            with open(os.path.join(directory, filename), 'rb') as file:
                circuit_data = pickle.load(file)
                if isinstance(circuit_data[0], tuple):
                    for circuit in circuit_data:
                        labels.append(circuit[1])
                        data.append([float(v) for k, v in circuit[0].items() if 'obs_' in k and v is not None])
                else:
                    continue

    if filename_to_save is None:
        raise ValueError("No files matched the condition.")

    return np.array(data), labels, filename_to_save

# Function to train Random Forest on data
def train_random_forest(data, labels, estimators, criterion, max_features):
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
    rfr = RandomForestRegressor(n_estimators=estimators, criterion=criterion, max_features=max_features, random_state=42)
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
            mean_absolute_error_train, mean_squared_error_train, root_mean_squared_error_train, r_squared_train, rfr)

def perform_pca(data, num_qubit, num_components=10):
    """
    Perform PCA on the given data and plot the explained variance ratio.
    
    Parameters:
    - data: The input data for PCA.
    - num_qubits: number of qubits of the quantum circuits analysed
    - num_components: Number of principal components to keep.
    
    Returns:
    - transformed_data: The data transformed into the principal component space.
    """
    ticks =2
    if num_qubit == 2:
        num_components = 13
        ticks = 1
    elif num_qubit == 3:
            num_components = 30
    elif num_qubit == 4:
            num_components = 55
    elif num_qubit == 5:
            num_components = 90
            ticks = 5

    pca = PCA(n_components=num_components)
    transformed_data = pca.fit_transform(data)
    # features = pca.components_
    # print(features[0])
    
    # Plot the explained variance ratio
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, num_components + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, num_components + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.xticks(ticks=np.arange(1, num_components + 1, ticks))  
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('PCA Explained Variance')
    plt.savefig(f"experiments/images/pca_{num_qubit}.png")
    
    return transformed_data

def hyperparameter_search(X_train, y_train):
    model = RandomForestRegressor(random_state=1)
    hp_grid = {
        'n_estimators': [250],  
        'max_features': ['sqrt'],#, 'log2'], 
        'criterion': ['squared_error', 'friedman_mse'],#, 'absolute_error'],
        # 'max_depth': [None, 10, 20, 30],
        # 'min_samples_split': [2, 5, 10],
        #'min_samples_leaf': [1, 2, 4]
    }
    GSCV = GridSearchCV(estimator=model, param_grid=hp_grid, cv=3)
    GSCV.fit(X_train, y_train)
    print("Best params:", GSCV.best_params_)
    return GSCV.best_params_

def save_dataframe(filename, model, pca, dataset, hyperparameters, metrics):
    """
    Save the model results to a .pkl file.

    Parameters:
    - filename: The name of the file to save the results.
    - model: The trained model.
    - pca: boolean. 
    - dataset: The dataset used for training.
    - hyperparameters: A dictionary of the model's hyperparameters.
    - metrics: A tuple of the model's performance metrics.
    """
    performance_metrics = {
        'MAE Test': metrics[0],
        'MSE Test': metrics[1],
        'RMSE Test': metrics[2],
        'R² Test': metrics[3],
        'MAE Train': metrics[4],
        'MSE Train': metrics[5],
        'RMSE Train': metrics[6],
        'R² Train': metrics[7]
    }
    result = {
        'Model': model,
        "pca": pca,
        'Dataset': dataset,
        'hyperparameters': hyperparameters,
        'performance_metrics': performance_metrics
    }

    # Create a DataFrame from the result
    result_df = pd.DataFrame([result])

    # Check if the file exists
    if os.path.exists(filename):
        # Load existing DataFrame
        existing_df = pd.read_pickle(filename)
        # Append the new result
        updated_df = pd.concat([existing_df, result_df], ignore_index=True)
    else:
        # If file doesn't exist, use the new DataFrame
        updated_df = result_df

    # Save the updated DataFrame to the file
    updated_df.to_pickle(filename)
    print(f"Data successfully saved in {filename}")

def save_model(dataset_name, pca, num_qubit, gates_set, data, labels):
    filename = f'experiments/results_{dataset_name}_qubit_{num_qubit}.pkl'
     # Get Results over the best hyperparameter found from Grid Search
    best_hp = hyperparameter_search(data, labels)
    results = train_random_forest(data, labels, estimators=best_hp["n_estimators"], criterion=best_hp["criterion"], max_features=best_hp["max_features"])
    dataset_limit = ""
    if dataset_name =="random":
        dataset_limit = "gates"
    elif dataset_name == "tim":
        dataset_limit ="trotter_steps"

    save_dataframe(filename=filename, model=results[-1], pca=pca, dataset=f"{dataset_limit}_{gates_set}", hyperparameters=best_hp, metrics=results[:-1])


# Main function to execute the process
def main():
    dataset_name = "tim"
    directory = 'data/dataset_'+dataset_name

    # Choose keywords to have in the dataset file to be studied
    #dataset_limited = [19,39,59]
    dataset_limited = [1,2,3,4,5]

    for num_qubit in range(2, 5):
        for training_choice in dataset_limited:
            data, labels, filename = load_data(directory, num_qubit, training_choice)
            if dataset_name=="random":
                gates_set = filename.split("gates_")[-1].split('.')[0]
                print(f"\Data size for qubit {num_qubit} and gates {gates_set}:", len(data), "Label size:", len(labels))
            else:
                gates_set = gates_set = filename.split("trotter_")[-1].split('.')[0]

            models = ["Random_Forest", "pca"]
            for model in models:
                pca = False
                if model == "pca":
                    data = perform_pca(data, num_qubit)
                    pca = True
                save_model(dataset_name, pca, num_qubit, gates_set, data, labels)
                 
if __name__ == "__main__":
    main()
