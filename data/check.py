import os

# Define the directory containing the files
directory = 'data/random_circuits'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Construct the new filename by removing the prefix
    new_filename = filename.replace('random_circuits_', '', 1)
    # Construct full file paths
    old_file = os.path.join(directory, filename)
    new_file = os.path.join(directory, new_filename)
    # Rename the file
    os.rename(old_file, new_file)
    print(f'Renamed: {old_file} to {new_file}')
