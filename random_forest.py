import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.decomposition import PCA

# Function to load data from .pkl files
def load_data(directory, num_qubit):
    labels, data = [], []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl') and "qubits_"+str(num_qubit) in filename:
            # print(filename)
            with open(os.path.join(directory, filename), 'rb') as file:
                circuit_data = pickle.load(file)
                if isinstance(circuit_data[0], tuple):
                    for circuit in circuit_data:
                        labels.append(circuit[1])
                        data.append([float(v) for k, v in circuit[0].items() if 'obs_' in k and v is not None])
                else:
                    continue
    return np.array(data), labels

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
    if num_qubit == 2:
        num_components = 13
    elif num_qubit == 3:
            num_components = 30
    elif num_qubit == 4:
            num_components = 55
    elif num_qubit == 5:
            num_components = 90

    pca = PCA(n_components=num_components)
    transformed_data = pca.fit_transform(data)
    
    # Plot the explained variance ratio
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, num_components + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, num_components + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.xticks(ticks=np.arange(1, num_components + 1, 1))  # Set x-axis ticks to integers
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('PCA Explained Variance')
    plt.savefig(f"pca_{num_qubit}.png")
    
    return transformed_data

def hyperparameter_search(X_train, y_train):
    model = RandomForestRegressor(random_state=1)
    hp_grid = {
        'n_estimators': [100, 200],  
        'max_features': ['sqrt'],  # Avoid 'auto' for small datasets
        'criterion': ['squared_error', 'friedman_mse']   # Avoid 'absolute_error' if unsupported
    }
    GSCV = GridSearchCV(estimator=model, param_grid=hp_grid, cv=3)
    GSCV.fit(X_train, y_train)
    # print("Best params:", GSCV.best_params_)
    return GSCV.best_params_

def save_model_results(filename, model, hyperparameters, metrics):
    """
    Save the model results to a .pkl file.

    Parameters:
    - filename: The name of the file to save the results.
    - model: The trained model.
    - hyperparameters: A dictionary of the model's hyperparameters.
    - performance_metrics: A dictionary of the model's performance metrics.
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
        'Random_Forest': model,
        'hyperparameters': hyperparameters,
        'performance_metrics': performance_metrics
    }
    
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as file:
        pickle.dump(result, file)
    print(f"Results saved to {filename}")


# Main function to execute the process
def main():
    dataset = "random"
    directory = 'data/'+dataset+'_circuits'

    results_dict, results_dict_pca = {}, {}
    for num_qubit in range(5, 6):  
        data, labels = load_data(directory, num_qubit)
        print(f"Data size for qubit {num_qubit}:", len(data), "Label size:", len(labels))
        
        # Create dictionary to save the results
        results_dict[num_qubit] = []

        """# Get Results over the best hyperparameter found from Grid Search
        best_hp = hyperparameter_search(data, labels)
        results = train_random_forest(data, labels, estimators=best_hp["n_estimators"], criterion=best_hp["criterion"], max_features=best_hp["max_features"])
        results_dict[num_qubit].append(results)
        save_model_results('experiments/results_'+dataset+f'_qubit_{num_qubit}.pkl', model=results[-1], hyperparameters=best_hp, metrics=results[:-1])"""

        # Same with PCA preselected feratures
        results_dict_pca[num_qubit] = [] 
        # Perform PCA on the data
        transformed_data = perform_pca(data, num_qubit)

        best_hp_pca = hyperparameter_search(transformed_data, labels)
        results_pca = train_random_forest(transformed_data, labels, estimators=best_hp_pca["n_estimators"], criterion=best_hp_pca["criterion"], max_features=best_hp_pca["max_features"]) 
        results_dict_pca[num_qubit].append(results_pca)  
        save_model_results('experiments/results_'+dataset+f'_pca_qubit_{num_qubit}.pkl', model=results_pca[-1], hyperparameters=best_hp_pca, metrics=results_pca[:-1])


if __name__ == "__main__":
    main()
