import os
import pickle

def read_and_check_files(input_directory):
    for filename in os.listdir(input_directory):
        # Check if the filename is between 'dataset_random' and 'dataset_tim'
        if 'dataset_random' < filename < 'dataset_tim':
            file_path = os.path.join(input_directory, filename)
            
            with open(file_path, 'rb') as file:
                try:
                    data = pickle.load(file)
                except (pickle.UnpicklingError, EOFError, TypeError):
                    print(f"Error reading {filename}")
                    continue

                # Check if the data is a list and the first item is a tuple of length 2
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple) and len(data[0]) == 2:
                    print(f"{filename} meets the criteria")
                else:
                    print(f"{filename} does not meet the criteria")

# Example usage
input_directory = 'path/to/your/input/directory'
read_and_check_files(input_directory)
