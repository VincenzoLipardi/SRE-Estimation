# Quantum Circuit Dataset Generation and Analysis

This project generates a dataset of random quantum circuits and calculates their classical shadows for machine learning purposes.

## Files

- `data_generation.py`: Generates random quantum circuits using Qiskit and saves them as a dataset.
- `feature_design.py`: Calculates classical shadows for the generated circuits using PennyLane.
- `quantum_circuits_dataset.pkl`: The generated dataset of quantum circuits.
- `quantum_circuits_dataset_with_shadows.pkl`: The dataset with added classical shadows.

## Requirements

- Python 3.7+
- Qiskit
- PennyLane
- NumPy
- Pandas
- Matplotlib
- NetworkX

## Usage

1. Run `data_generation.py` to generate the initial dataset of quantum circuits.
2. Run `feature_design.py` to calculate classical shadows and add them to the dataset.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
