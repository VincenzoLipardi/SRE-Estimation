# A study on Classical Representation of Quantum Circuits for Machine Learning 

In this project we study different classical representation of quantum circuits in order to train machine learning models to predict quantum circuits properties.

## Tutorials

To help you get started with the dataset and the machine learning models, we provide tutorial notebooks:

- [Data Tutorial](data/tutorial_data.ipynb): This Jupyter notebook walks you through the process of generating the dataset, calculating classical shadows, and training machine learning models. It serves as a practical guide to understanding and utilizing the resources in this repository.

## Files

### Data Folder (`data/`)

The `data/` folder contains the core scripts for generating, processing, and labeling quantum circuit datasets:

- **`circuits_generation.py`**: Generates two types of quantum circuits using PennyLane and Qiskit:
  - **Random circuits**: Creates circuits with random gates from specified basis sets (Clifford+T, Clifford, or rotations+CX) with configurable qubit counts (2-6) and gate counts (0-99 gates in ranges)
  - **Ising model circuits**: Generates Trotterized transverse-field Ising model circuits with configurable Trotter steps and random RX/RZ angles
  - Outputs circuits in QASM format and saves them as pickle files

- **`features.py`**: Computes classical representations and features from quantum circuits:
  - **Classical shadows**: Calculates expectation values of Pauli observables using the classical shadow protocol (main quantum feature extraction method)
  - **Gate binning**: Extracts gate counts and bins rotation angles into histograms for classical circuit analysis
  - **Extended gate features**: Combines gate binning with one-hot encoded qubit count representations
  - Processes existing circuit files and adds computed features as new columns

- **`create_dataset.py`**: Creates machine learning-ready datasets by extracting specific feature sets:
  - Organizes different feature representations (shadows, gate_bins, gate_bins_ext, combined) into separate subfolder datasets
  - Converts circuit dictionaries into feature vectors paired with labels for ML training

- **`label.py`**: Computes quantum circuit complexity labels using stabilizer Rényi entropy:
  - Calculates the α=2 stabilizer Rényi entropy as a measure of quantum circuit "magic" (non-stabilizer resource)
  - Processes circuit datasets and adds entropy values as target labels for supervised learning

- **`data_distribution.py`**: Provides data analysis and visualization tools:
  - Loads processed datasets and extracts entropy distributions
  - Creates histograms showing entropy distributions across different circuit parameters (qubit counts, gate counts, Trotter steps)



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
4. Run [create_dataset.py](lcreate_dataset.py) to create the dataset of quantum circuits with several classical representations.

Train Machine Learning models:

- [classical ML models](models.py)
- [Graph Neural Network (GNN)](gnn.py)
- [Training Pipeline](training.py)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
