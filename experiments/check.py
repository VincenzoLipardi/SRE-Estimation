import pickle
import pandas as pd

def check_rf_results(file_path):
    """
    Opens and examines the contents of a specific Random Forest results file,
    removes duplicates, and saves the deduplicated DataFrame back to the same file.
    """
    df = pd.read_pickle(file_path)
    # Select columns for duplicate checking
    selected_columns = ['Dataset', 'Dataset_limited', 'features']
    
    # Remove duplicates based on selected columns, keeping the first occurrence
    df = df.drop_duplicates(subset=selected_columns, keep='first')
    
    # Save the deduplicated DataFrame back to the same file
    df.to_pickle(file_path)
    print(f"Duplicates removed and saved to {file_path}")

def print_dataframe(file_path):
    """
    Prints the DataFrame showing only selected columns.
    
    Args:
        file_path (str): Path to the pickle file containing the DataFrame
    """
    df = pd.read_pickle(file_path)
    # print(df.columns)
    selected_columns = ['Dataset', 'features', "Model"]
    print(f"\nResults for {qubit} qubits:")
    print(df["Model"][1])
    print(df[selected_columns])

def find_rf_models(file_path):
    """
    Finds and displays rows that contain Random Forest models in the DataFrame.
    
    Args:
        file_path (str): Path to the pickle file containing the DataFrame
    """
    df = pd.read_pickle(file_path)
    # Filter for rows where Model is a RandomForestRegressor
    rf_rows = df[df['Model'].apply(lambda x: str(x).startswith('RandomForestRegressor'))]
    
    if len(rf_rows) == 0:
        print("No Random Forest models found in the dataset.")
        return
    
    print(f"\nFound {len(rf_rows)} Random Forest models:")
    selected_columns = ['Dataset', 'features', 'Model']
    print(rf_rows[selected_columns])
    return rf_rows

if __name__ == "__main__":
    qubit = 6
    file_path = f'experiments/results_qubit_{qubit}.pkl'
    #check_rf_results(file_path)
    
    #print_dataframe(file_path)
    find_rf_models(file_path)
