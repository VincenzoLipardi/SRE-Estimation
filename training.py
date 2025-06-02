from models import hyperparameter_search_rf, random_forest, hyperparameter_search_svm, svm, perform_pca, evaluate_model
import pandas as pd
import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Union, Any
import logging
from pathlib import Path
import sklearn.model_selection


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(
    directory: str, 
    num_qubit: int, 
    gate_filter_value: Union[bool, int, List[int]],
    reduce_data: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from the feature-specific directory for a given number of qubits and gate filter.
    
    Parameters:
        directory (str): Path to the feature-specific directory.
        num_qubit (int): Number of qubits.
        gate_filter_value (Union[bool, int, List[int]]): Filter condition for the dataset. Can be False (all), int, or list of ints (for gate/trotter ranges).
        reduce_data (bool): Whether to limit to the first 1000 items (default True).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels arrays.
    
    Raises:
        FileNotFoundError: If directory doesn't exist.
        ValueError: If no valid data files are found.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    labels, features = [], []
    pattern = f"qubits_{num_qubit}"
    
    for filename in os.listdir(directory):
        condition = (filename.endswith('.pkl') and pattern in filename)
        if gate_filter_value:
            if isinstance(gate_filter_value, list):
                # Match if any value in the list is in the filename
                condition = condition and any(str(val) in filename for val in gate_filter_value)
            else:
                condition = condition and str(gate_filter_value) in filename
        
        if condition:
            logger.info(f"Processing file: {filename}")
            try:
                file_path = os.path.join(directory, filename)
                with open(file_path, 'rb') as file:
                    circuit_data = pickle.load(file)
                    
                    if isinstance(circuit_data[0], tuple):
                        for feature, label in circuit_data:
                            if isinstance(feature, dict):
                                feature = list(feature.values())
                            features.append(feature)
                            labels.append(label)
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                continue
    
    if not features or not labels:
        raise ValueError(f"No valid data found in {directory} for {num_qubit} qubits")
    
    # Optionally limit to first 1000 items
    if reduce_data:
        features = features[:1000]
        labels = labels[:1000]

    # Convert to numpy arrays and ensure proper shape
    features_array = np.array(features).reshape(-1, len(features[0]))
    labels_array = np.array(labels).flatten()
    
    return features_array, labels_array

def save_model(
    model: str,
    dataset: str,
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    gate_filter_value: Union[bool, int],
    num_qubit: int,
    features_type: str,
    pca: bool,
    verbose: bool,
    results_subdir: str = ''
) -> None:
    """
    Train and save model with best hyperparameters for the given configuration.
    
    Parameters:
        model (str): Model type ('rfr' or 'svr').
        dataset (str): Dataset name.
        features (np.ndarray): Input features.
        labels (np.ndarray): Target labels.
        test_size (float): Test set proportion.
        gate_filter_value (Union[bool, int, str]): Filter condition for the dataset.
        num_qubit (int or str): Number of qubits or 'all'.
        features_type (str): Type of features.
        pca (bool): Whether PCA was applied.
        verbose (bool): Verbosity flag.
        results_subdir (str): Subdirectory under 'experiments' to save results.
    Raises:
        ValueError: If unsupported model type is provided.
    """
    results_dir = Path('experiments') / results_subdir if results_subdir else Path('experiments')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    filename = results_dir / f'results_{model}_qubit_{num_qubit}.pkl'
    
    try:
        # Ensure labels are numpy arrays
        labels = np.array([np.array(label) if not isinstance(label, np.ndarray) else label for label in labels])
        
        if model == "rfr":
            best_hp = hyperparameter_search_rf(features, labels)
            results = random_forest(
                features, labels, test_size, verbose,
                estimators=best_hp["n_estimators"],
                criterion=best_hp["criterion"],
                max_features=best_hp["max_features"]
            )
        elif model == "svr":
            best_hp = hyperparameter_search_svm(features, labels)
            results = svm(
                features, labels, test_size, verbose,
                kernel=best_hp["kernel"],
                C=best_hp["C"],
                epsilon=best_hp["epsilon"]
            )
        else:
            raise ValueError(f"Unsupported model type: {model}")

        gate_filter_value = "all" if gate_filter_value is False else gate_filter_value
        save_dataframe(
            filename=filename,
            model=results[-1],
            features=features_type,
            pca=pca,
            dataset=dataset,
            gate_filter_value=gate_filter_value,
            hyperparameters=best_hp,
            metrics=results[:-1]
        )
    except Exception as e:
        logger.error(f"Error in save_model: {str(e)}")
        raise

def save_dataframe(
    filename: Union[str, Path],
    model: Any,
    features: str,
    pca: bool,
    dataset: str,
    gate_filter_value: Union[str, int],
    hyperparameters: Dict[str, Any],
    metrics: Tuple[float, ...]
) -> None:
    """
    Save model results to a pickle file, appending if the file exists.
    
    Parameters:
        filename (Union[str, Path]): Output file path.
        model (Any): Trained model.
        features (str): Feature type.
        pca (bool): PCA flag.
        dataset (str): Dataset name.
        gate_filter_value (Union[str, int]): Filter condition for the dataset.
        hyperparameters (Dict[str, Any]): Model hyperparameters.
        metrics (Tuple[float, ...]): Performance metrics (MAE, MSE, RMSE, R2 for test/train).
    """
    performance_metrics = {
        'Mean Absolute Error Test': metrics[0],
        'MSE Test': metrics[1],
        'R² Test': metrics[2],
        'Mean Absolute Error Train': metrics[3],
        'MSE Train': metrics[4],
        'R² Train': metrics[5]
    }
    
    result = {
        'Dataset': dataset,
        'Dataset_limited': gate_filter_value,
        'Model': model,
        'features': features,
        'pca': pca,
        'hyperparameters': hyperparameters,
        'performance_metrics': performance_metrics
    }

    try:
        result_df = pd.DataFrame([result])
        
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            existing_df = pd.read_pickle(filename)
            updated_df = pd.concat([existing_df, result_df], ignore_index=True)
        else:
            updated_df = result_df

        updated_df.to_pickle(filename)
        logger.info(f"Data successfully saved in {filename}")
        
    except Exception as e:
        logger.error(f"Error saving results to {filename}: {str(e)}")
        raise

def train_and_save(
    dataset: str,
    models: List[str],
    test_size: float,
    qubit_min: int,
    qubit_max: int,
    feature_type: str,
    qubit_mode: str,
    gate_mode: str,
    gate_filter: List[int] = None,
    pca: bool = False,
    verbose: bool = False,
    results_subdir: str = '',
    reduce_data: bool = False
) -> None:
    """
    Train and save models for different configurations of qubits and gates.
    
    Parameters:
        dataset (str): Name of the dataset.
        models (List[str]): List of model types to train ('rfr', 'svr').
        test_size (float): Proportion of data to use for testing.
        qubit_min (int): Minimum number of qubits to include.
        qubit_max (int): Maximum number of qubits to include (exclusive).
        feature_type (str): Type of features to use.
        qubit_mode (str): 'independent' (train/test per qubit) or 'all' (combine all qubits).
        gate_mode (str): 'all' (use all gates) or 'specific' (filter by gate_filter).
        gate_filter (List[int], optional): List of gate numbers or ranges if gate_mode is 'specific'.
        pca (bool, optional): Whether to apply PCA to features.
        verbose (bool, optional): Verbosity flag.
        results_subdir (str): Subdirectory under 'experiments' to save results.
        reduce_data (bool): Whether to limit to the first 1000 items (default True).
    Raises:
        FileNotFoundError: If the data directory does not exist.
        ValueError: If invalid mode or missing filter is provided.
    """
    directory = Path(f'data/dataset_{dataset}/{feature_type}')
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")

    # Determine gate numbers to use
    if gate_mode == "all":
        gate_filter = [False]
    elif gate_mode == "specific" and gate_filter is not None:
        pass
    else:
        raise ValueError("Invalid gate_mode or missing gate_filter")

    if qubit_mode == "independent":
        for num_qubit in range(qubit_min, qubit_max):
            for gate_filter_value in gate_filter:
                try:
                    features, labels = load_data(str(directory), num_qubit, gate_filter_value, reduce_data=reduce_data)
                    for model in models:
                        if pca:
                            current_features = perform_pca(dataset, features, num_qubit)
                        else:
                            current_features = features.copy()
                        save_model(
                            model, dataset, current_features, labels,
                            test_size, gate_filter_value, num_qubit,
                            feature_type, pca, verbose,
                            results_subdir=results_subdir
                        )
                except Exception as e:
                    logger.error(f"Error processing qubit {num_qubit}, training_choice {gate_filter_value}: {str(e)}")
                    continue
    elif qubit_mode == "all":
        all_features, all_labels = [], []
        for num_qubit in range(qubit_min, qubit_max):
            for gate_filter_value in gate_filter:
                try:
                    features, labels = load_data(str(directory), num_qubit, gate_filter_value, reduce_data=reduce_data)
                    all_features.append(features)
                    all_labels.append(labels)
                except Exception as e:
                    logger.error(f"Error loading data for qubit {num_qubit}, training_choice {gate_filter_value}: {str(e)}")
                    continue
        if all_features and all_labels:
            features = np.concatenate(all_features, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            for model in models:
                pca = False
                current_features = features.copy()
                if model == "PCA":
                    current_features = perform_pca(dataset, current_features, None)
                    pca = True
                save_model(
                    model, dataset, current_features, labels,
                    test_size, "all", "all",
                    feature_type, pca, verbose,
                    results_subdir=results_subdir
                )
    else:
        raise ValueError("Invalid qubit_mode")


def train_generalization_qubits(
    dataset: str,
    model: str,
    feature_type: str,
    train_qubits: list,
    test_qubit: int,
    gate_mode: str,
    gate_filter: list = None,
    verbose: bool = False
) -> None:
    """
    Train on all train_qubits, test on test_qubit, and report metrics for both cumulative and external test sets.

    Parameters:
        dataset (str): Dataset name.
        model (str): Model type ('rfr' or 'svr').
        feature_type (str): Feature type.
        train_qubits (list): List of qubits to use for training.
        test_qubit (int): Qubit to use for external testing.
        gate_mode (str): 'all' or 'specific'.
        gate_filter (list, optional): List of gate numbers/ranges if gate_mode is 'specific'.
        verbose (bool, optional): Verbosity flag.

    Notes:
        - Trains on all train_qubits, tests on test_qubit.
        - Reports both internal (split) and external (held-out qubit) test metrics.
    """

    directory = Path(f'data/dataset_{dataset}/{feature_type}')
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")

    # Determine gate numbers to use
    if gate_mode == "all":
        gate_filter = [False]
    elif gate_mode == "specific" and gate_filter is not None:
        pass
    else:
        raise ValueError("Invalid gate_mode or missing gate_filter")

    # Load and concatenate training data (qubits 2-5)
    all_features, all_labels = [], []
    for num_qubit in train_qubits:
        for gate_filter_value in gate_filter:
            try:
                features, labels = load_data(str(directory), num_qubit, gate_filter_value)
                all_features.append(features)
                all_labels.append(labels)
            except Exception as e:
                logger.error(f"Error loading data for qubit {num_qubit}, training_choice {gate_filter_value}: {str(e)}")
                continue
    if not all_features or not all_labels:
        logger.error("No training data loaded for cumulative training.")
        return
    X_train_all = np.concatenate(all_features, axis=0)
    y_train_all = np.concatenate(all_labels, axis=0)

    # Load test data (qubit 6)
    X_test_ext, y_test_ext = None, None
    for gate_filter_value in gate_filter:
        try:
            X_test_ext, y_test_ext = load_data(str(directory), test_qubit, gate_filter_value)
            break
        except Exception as e:
            logger.error(f"Error loading data for test qubit {test_qubit}, training_choice {gate_filter_value}: {str(e)}")
            continue
    if X_test_ext is None or y_test_ext is None:
        logger.error("No external test data loaded for test qubit.")
        return

    # Train and evaluate model on cumulative set
    if model == "rfr":
        best_hp = hyperparameter_search_rf(X_train_all, y_train_all)
        results = random_forest(
            X_train_all, y_train_all, 0.2, verbose,
            estimators=best_hp["n_estimators"],
            criterion=best_hp["criterion"],
            max_features=best_hp["max_features"]
        )
        trained_model = results[-1]
        metrics_train_test = results[:-1]
    elif model == "svr":
        best_hp = hyperparameter_search_svm(X_train_all, y_train_all)
        results = svm(
            X_train_all, y_train_all, 0.2, verbose,
            kernel=best_hp["kernel"],
            C=best_hp["C"],
            epsilon=best_hp["epsilon"]
        )
        trained_model = results[-1]
        metrics_train_test = results[:-1]
    else:
        raise ValueError(f"Unsupported model type: {model}")

    # Evaluate on external test set (qubit 6)
    ext_mae, ext_mse, ext_r2 = evaluate_model(trained_model, X_test_ext, y_test_ext)
    logger.info(f"External test (qubit {test_qubit}) metrics: MAE={ext_mae}, MSE={ext_mse}, R2={ext_r2}")

    # Save all results
    results_dir = Path('experiments/extrapolation')
    results_dir.mkdir(exist_ok=True)
    filename = results_dir / f'results_{model}_train_qubits_{"-".join(map(str, train_qubits))}_test_qubit_{test_qubit}.pkl'

    performance_metrics = {
        'Mean Absolute Error Test': metrics_train_test[0],
        'MSE Test': metrics_train_test[1],
        'R² Test': metrics_train_test[2],
        'Mean Absolute Error Train': metrics_train_test[3],
        'MSE Train': metrics_train_test[4],
        'R² Train': metrics_train_test[5],
        'Mean Absolute Error External Test': ext_mae,
        'MSE External Test': ext_mse,
        'R² External Test': ext_r2
    }
    result = {
        'Dataset': dataset,
        'Train Qubits': train_qubits,
        'Test Qubit': test_qubit,
        'Model': trained_model,
        'features': feature_type,
        'hyperparameters': best_hp,
        'performance_metrics': performance_metrics
    }
    try:
        result_df = pd.DataFrame([result])
        if filename.exists() and filename.stat().st_size > 0:
            existing_df = pd.read_pickle(filename)
            updated_df = pd.concat([existing_df, result_df], ignore_index=True)
        else:
            updated_df = result_df
        updated_df.to_pickle(filename)
        logger.info(f"Cumulative and external test results saved in {filename}")
    except Exception as e:
        logger.error(f"Error saving cumulative results to {filename}: {str(e)}")
        raise

def train_generalization_gates(model, datasets, feature_type, test_size, verbose=True):
    """
    Perform generalization study for datasets 'random' and 'tim' using both Random Forest and SVM.
    Trains on specified qubits and gate/trotter ranges, evaluates on both split test set and unseen circuits.
    Saves both results in the same file per dataset.

    Parameters:
        model (str): Model type ('rfr' or 'svr').
        datasets (list): List of dataset names ('random', 'tim').
        feature_type (str): Feature type.
        test_size (float): Proportion of data to use for test split.
        verbose (bool, optional): Verbosity flag.

    Notes:
        - For 'random': trains on gate ranges 0-79, tests on 80-99.
        - For 'tim': trains on trotter steps 1-4, tests on 5.
        - Both train/test split and extrapolation (unseen gates/steps) are evaluated.
    """
    results_dir = Path('experiments/extrapolation')
    results_dir.mkdir(exist_ok=True)
    test_size = 0.2
    verbose = True
    for dataset in datasets:
        logger.info(f"Generalization study for dataset: {dataset}")
        directory = Path(f'data/dataset_{dataset}/{feature_type}')
        if not directory.exists():
            logger.error(f"Directory {directory} does not exist")
            continue
        qubits = list(range(2, 7))
        results_list = []
        if dataset == "random":
            # Train: files with gates_0-19, 20-39, 40-59, 60-79; Test: gates_80-99
            train_gate_ranges = ["0-19", "20-39", "40-59", "60-79"]
            unseen_gate_ranges = ["80-99"]
            # Load training data
            all_features, all_labels = [], []
            for num_qubit in qubits:
                try:
                    features, labels = load_data(str(directory), num_qubit, train_gate_ranges)
                    all_features.append(features)
                    all_labels.append(labels)
                except Exception as e:
                    logger.error(f"Error loading training data for qubit {num_qubit}: {str(e)}")
            if not all_features or not all_labels:
                logger.error("No training data loaded for generalization study (random).")
                continue
            X, y = np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)
            # Split train/test
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                X, y, test_size=test_size, random_state=42)
            # Models to run
            
            if model == "rfr":
                best_hp = hyperparameter_search_rf(X_train, y_train)
                results = random_forest(
                    X_train, y_train, test_size=test_size, verbose=verbose,
                    estimators=best_hp["n_estimators"],
                    criterion=best_hp["criterion"],
                    max_features=best_hp["max_features"]
                )
            elif model == "svr":
                best_hp = hyperparameter_search_svm(X_train, y_train)
                results = svm(
                    X_train, y_train, test_size=test_size, verbose=verbose,
                    kernel=best_hp["kernel"],
                    C=best_hp["C"],
                    epsilon=best_hp["epsilon"]
                )
            trained_model = results[-1]
            # Evaluate on train and test split
            train_metrics = evaluate_model(trained_model, X_train, y_train)
            test_metrics = evaluate_model(trained_model, X_test, y_test)
            # Evaluate on unseen gates
            unseen_features, unseen_labels = [], []
            for num_qubit in qubits:
                try:
                    f, l = load_data(str(directory), num_qubit, unseen_gate_ranges)
                    unseen_features.append(f)
                    unseen_labels.append(l)
                except Exception as e:
                    logger.warning(f"No unseen data for qubit {num_qubit}: {str(e)}")
            if unseen_features and unseen_labels:
                X_unseen = np.concatenate(unseen_features, axis=0)
                y_unseen = np.concatenate(unseen_labels, axis=0)
                # Sample 20% of the unseen data randomly
                if len(X_unseen) > 0:
                    X_unseen, _, y_unseen, _ = sklearn.model_selection.train_test_split(
                        X_unseen, y_unseen, test_size=0.8, random_state=42
                    )
                unseen_metrics = evaluate_model(trained_model, X_unseen, y_unseen)
            else:
                unseen_metrics = (None, None, None, None)
            # Save results for this model
            result = {
                'Dataset': dataset,
                'Model Type': model,
                'Train Qubits': qubits,
                'Train Gates': train_gate_ranges,
                'Test Gates': unseen_gate_ranges,
                'Model': trained_model,
                'features': feature_type,
                'hyperparameters': best_hp,
                'Train Metrics (MAE, MSE, RMSE, R2)': train_metrics,
                'Test Split Metrics (MAE, MSE, RMSE, R2)': test_metrics,
                'Unseen Metrics (MAE, MSE, RMSE, R2)': unseen_metrics
            }
            results_list.append(result)
            filename = results_dir / f'{model}_depth_generalization_{dataset}.pkl'
            pd.DataFrame(results_list).to_pickle(filename)
            logger.info(f"Generalization results saved in {filename}")
        elif dataset == "tim":
            # Train: trotter steps 1-4, Test: step 5
            train_steps = [f"trotter_{i}" for i in range(1, 5)]
            unseen_steps = ["trotter_5"]
            all_features, all_labels = [], []
            for num_qubit in qubits:
                try:
                    features, labels = load_data(str(directory), num_qubit, train_steps)
                    all_features.append(features)
                    all_labels.append(labels)
                except Exception as e:
                    logger.error(f"Error loading training data for qubit {num_qubit}: {str(e)}")
            if not all_features or not all_labels:
                logger.error("No training data loaded for generalization study (tim).")
                continue
            X, y = np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)
            # Split train/test
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                X, y, test_size=test_size, random_state=42)
            
            if model == "rfr":
                best_hp = hyperparameter_search_rf(X_train, y_train)
                results = random_forest(
                    X_train, y_train, test_size=test_size, verbose=verbose,
                    estimators=best_hp["n_estimators"],
                    criterion=best_hp["criterion"],
                    max_features=best_hp["max_features"]
                )
            elif model == "svr":
                best_hp = hyperparameter_search_svm(X_train, y_train)
                results = svm(
                    X_train, y_train, test_size=test_size, verbose=verbose,
                    kernel=best_hp["kernel"],
                    C=best_hp["C"],
                    epsilon=best_hp["epsilon"]
                )
            trained_model = results[-1]
            # Evaluate on train and test split
            train_metrics = evaluate_model(trained_model, X_train, y_train)
            test_metrics = evaluate_model(trained_model, X_test, y_test)
            # Evaluate on unseen trotter step
            unseen_features, unseen_labels = [], []
            for num_qubit in qubits:
                try:
                    f, l = load_data(str(directory), num_qubit, unseen_steps)
                    unseen_features.append(f)
                    unseen_labels.append(l)
                except Exception as e:
                    logger.warning(f"No unseen data for qubit {num_qubit}: {str(e)}")
            if unseen_features and unseen_labels:
                X_unseen = np.concatenate(unseen_features, axis=0)
                y_unseen = np.concatenate(unseen_labels, axis=0)
                # Sample 20% of the unseen data randomly
                if len(X_unseen) > 0:
                    X_unseen, _, y_unseen, _ = sklearn.model_selection.train_test_split(
                        X_unseen, y_unseen, test_size=0.8, random_state=42
                    )
                unseen_metrics = evaluate_model(trained_model, X_unseen, y_unseen)
            else:
                unseen_metrics = (None, None, None, None)
            # Save results for this model
            result = {
                'Dataset': dataset,
                'Model Type': model,
                'Train Qubits': qubits,
                'Train Trotter Steps': train_steps,
                'Test Trotter Steps': unseen_steps,
                'Model': trained_model,
                'features': feature_type,
                'hyperparameters': best_hp,
                'Train Metrics (MAE, MSE, RMSE, R2)': train_metrics,
                'performance test ': test_metrics,
                'performance extrapolation ': unseen_metrics
            }
            results_list.append(result)
            filename = results_dir / f'{model}_depth_generalization_{dataset}.pkl'
            pd.DataFrame(results_list).to_pickle(filename)
            logger.info(f"Generalization results saved in {filename}")

def run_standard_independent_training(datasets, feature_types, models, test_size, qubit_min, qubit_max, qubit_mode, gate_mode, gate_filter, pca, verbose, results_subdir='', reduce_data=False):
    """
    Run standard independent training for all combinations of datasets, feature_types, and models.

    Parameters:
        datasets (list): List of dataset names.
        feature_types (list): List of feature types.
        models (list): List of model types ('rfr', 'svr').
        test_size (float): Proportion of data to use for testing.
        qubit_min (int): Minimum number of qubits.
        qubit_max (int): Maximum number of qubits (exclusive).
        qubit_mode (str): 'independent' or 'all'.
        gate_mode (str): 'all' or 'specific'.
        gate_filter (list): List of gate numbers/ranges if gate_mode is 'specific'.
        pca (bool): Whether to apply PCA.
        verbose (bool): Verbosity flag.
        results_subdir (str): Subdirectory under 'experiments' to save results.
        reduce_data (bool): Whether to limit to the first 1000 items (default False).
    """
    for dataset in datasets:
        for feature_type in feature_types:

            logger.info(f"Training models for dataset: {dataset}, feature type: {feature_type}")
            train_and_save(
                dataset=dataset,
                models=models,
                test_size=test_size,
                qubit_min=qubit_min,
                qubit_max=qubit_max,
                feature_type=feature_type,
                qubit_mode=qubit_mode,
                gate_mode=gate_mode,
                gate_filter=gate_filter,
                pca=pca,
                verbose=verbose,
                results_subdir=results_subdir,
                reduce_data=reduce_data
            )

def run_generalization_qubits(datasets, feature_type, model, train_qubits, test_qubit, gate_mode, gate_filter, verbose):
    """
    Run cumulative training for the specified datasets and parameters.

    Parameters:
        datasets (list): List of dataset names.
        feature_type (str): Feature type.
        model (str): Model type ('rfr', 'svr').
        train_qubits (list): List of qubits to use for training.
        test_qubit (int): Qubit to use for external testing.
        gate_mode (str): 'all' or 'specific'.
        gate_filter (list): List of gate numbers/ranges if gate_mode is 'specific'.
        verbose (bool): Verbosity flag.
    """
    for dataset in datasets:
        logger.info(f"Cumulative training for dataset: {dataset}, feature: {feature_type}, train qubits {train_qubits}, test qubit {test_qubit}")
        train_generalization_qubits(
            dataset=dataset,
            model=model,
            feature_type=feature_type,
            train_qubits=train_qubits,
            test_qubit=test_qubit,
            gate_mode=gate_mode,
            gate_filter=gate_filter,
            verbose=verbose
        )

if __name__ == "__main__":
    # Main experiment configuration
    datasets = ["random", "tim"]
    feature_types = ["gate_bins", "shadow", "combined"]
    models = ["svr", "rfr"]
    test_size = 0.2
    qubit_min = 2
    qubit_max = 7
    qubit_mode = "independent"  # Choose between "independent" (per qubit) or "all" (combine all qubits)
    gate_mode = "all"            # Choose between "all" (all gates) or "specific" (filtered by gate_filter)
    gate_filter = [10, 20]     # Example: [10, 20], only used if gate_mode == "specific"
    pca=False

    # To run standard independent training, uncomment below:
    """for test_size in [0.2, 0.3, 0.4, 0.5, 0.1]:
        run_standard_independent_training(
            datasets=datasets,
            feature_types=feature_types,
            models=models,
            test_size=test_size,
            qubit_min=qubit_min,
            qubit_max=qubit_max,
            qubit_mode=qubit_mode,
            gate_mode=gate_mode,
            gate_filter=gate_filter,
            pca=pca,
            verbose=True,
            results_subdir=str(test_size),
            reduce_data=True
        )
    
    # To run generalization study (depth/trotter extrapolation), run:
    for model in models:
        run_generalization_qubits(
        datasets=datasets,
        feature_type="gate_bins_ext",
        model=model,
        train_qubits=[2, 3, 4, 5],
        test_qubit=6,
        gate_mode=gate_mode,
        gate_filter=gate_filter,
        verbose=True
    )"""
        
    for model in models:
        train_generalization_gates(
            model=model,
            datasets=datasets,
            feature_type="gate_bins_ext",
            test_size=test_size,
            verbose=True
        )
        