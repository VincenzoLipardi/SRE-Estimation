# A study on Classical Representation of Quantum Circuits for Machine Learning 

In this project we study different classical representation of quantum circuits in order to train machine learning models to predict quantum circuits properties.

## Tutorials

To help you get started with the dataset and the machine learning models, we provide tutorial notebooks:

- [Data Tutorial](data/tutorial_data.ipynb): This Jupyter notebook walks you through the process of generating the dataset, calculating classical shadows, and training machine learning models. It serves as a practical guide to understanding and utilizing the resources in this repository.

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

Generate data if they do not appear in [directory](data/random_circuits/)
1. Run [circuit_generation.py](circuit_generation.py) to generate the initial dataset of quantum circuits.
2. Run [feature.py](feature.py) to calculate classical shadows as classical features and add them to the dataset.
3. Run [label.py](label.py) to label the quantum circuits of the dataset.

Train Machine Learning models:
- [Random Forests](random_forest.py)
- [Support Vector Machine (SVM)](svm.py)
- [Graph Neural Network (GNN)](gnn.py)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
