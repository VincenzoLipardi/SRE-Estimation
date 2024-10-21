# Quantum Circuit Dataset Generation and Analysis

In this project we train machine learning models on quantum circuit objects to estimate quantum magic and mitigate noisy expectation values.
You can find the code used to generate the dataset of random quantum circuits with their classical shadows, as we;; as the machine learning models designed.

## Files
Folder data
- `data_generation.py`: Generates random quantum circuits using Qiskit and saves them as a dataset.
- `feature_design.py`: Calculates classical shadows for the generated circuits using PennyLane.
- `.pkl`: The generated dataset of quantum circuits indicating the type of gates used, the number of qubits and the range of total gates.


## Requirements

- Python 3.7+
Install [requirements](requirements.txt)

Main Libraries needed:
    - Qiskit
    - PennyLane
    - NumPy
    - Pandas
    - Matplotlib
    - NetworkX

## Usage

Generate data if they do not appear in [directory](data)
1. Run [file](data_generation.py) to generate the initial dataset of quantum circuits.
2. Run [file](feature.py)to calculate classical shadows and add them to the dataset.

Train Machine Learnign model:
[Random Forests](random_forest.py) 

## License

This project is licensed under the MIT License - see the LICENSE file for details.
