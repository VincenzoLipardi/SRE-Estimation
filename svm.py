import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from random_forest import load_data, save_dataframe

def train_svm(data, labels, kernel='rbf', C=1.0, epsilon=0.1):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
    svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    # Calculate metrics for test set
    mean_absolute_error_test = mean_absolute_error(y_test, y_pred)
    mean_squared_error_test = mean_squared_error(y_test, y_pred)
    root_mean_squared_error_test = np.sqrt(mean_squared_error_test)
    r_squared_test = r2_score(y_test, y_pred)

    # Calculate metrics for training set
    y_train_predictions = svr.predict(X_train)
    mean_absolute_error_train = mean_absolute_error(y_train, y_train_predictions)
    mean_squared_error_train = mean_squared_error(y_train, y_train_predictions)
    root_mean_squared_error_train = np.sqrt(mean_squared_error_train)
    r_squared_train = r2_score(y_train, y_train_predictions)

    print("\nSVM Training Set:")
    print(f"Mean Absolute Error: {mean_absolute_error_train:.4f}")
    print(f"Mean Squared Error: {mean_squared_error_train:.4f}")
    print(f"Root Mean Squared Error: {root_mean_squared_error_train:.4f}")
    print(f"R-squared Score: {r_squared_train:.4f}")

    print("SVM Test Set:")
    print(f"Mean Absolute Error: {mean_absolute_error_test:.4f}")
    print(f"Mean Squared Error: {mean_squared_error_test:.4f}")
    print(f"Root Mean Squared Error: {root_mean_squared_error_test:.4f}")
    print(f"R-squared Score: {r_squared_test:.4f}")
    return (mean_absolute_error_test, mean_squared_error_test, root_mean_squared_error_test, r_squared_test,
            mean_absolute_error_train, mean_squared_error_train, root_mean_squared_error_train, r_squared_train, svr)

def hyperparameter_search(X_train, y_train):
    model = SVR()
    hp_grid = {
        'C': [1,],  
              'gamma': [1], 
              'kernel': ['rbf']  
    }
    GSCV = GridSearchCV(estimator=model, param_grid=hp_grid, refit=True, cv=3)
    GSCV.fit(X_train, y_train)
    print("Best params:", GSCV.best_params_)
    return GSCV.best_params_

def main():
    dataset = "random"
    directory = 'data/'+dataset+'_circuits'

    results_dict = {}
    for num_qubit in range(2,7):  
        data, labels = load_data(directory, num_qubit)
        print(f"Data size for qubit {num_qubit}:", len(data), "Label size:", len(labels))
        
        # Create dictionary to save the results
        results_dict[num_qubit] = []

        # Get Results over the best hyperparameter found from Grid Search
        best_hp = hyperparameter_search(data, labels)
        results = train_svm(data, labels, kernel=best_hp["kernel"], C=best_hp["C"], epsilon=best_hp["gamma"])
        results_dict[num_qubit].append(results)
        save_dataframe('experiments/results_'+dataset+f'_svm_qubit_{num_qubit}.pkl', model=results[-1], hyperparameters= "default", metrics=results[:-1])

        

if __name__ == "__main__":
    main()
