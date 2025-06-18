#!/usr/bin/env python3
"""
Train the quantum circuit GNN on two large datasets:
1. data/dataset_tim (Ising model circuits)
2. data/dataset_random (Random basis rotation circuits)
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
import numpy as np
from data.dag_representation import QuantumCircuitDAGDataset
from typing import List
import os
import glob
import pandas as pd
import time
import pickle
from collections import defaultdict

class QuantumCircuitGNN(torch.nn.Module):
    """
    A Graph Neural Network for quantum circuit regression using Transformer convolutions and global pooling.
    Fixed version to avoid tensor mismatch issues.
    """
    
    def __init__(self, num_node_features: int = 5, heads: int = 4, hidden_dim: int = 128):
        super(QuantumCircuitGNN, self).__init__()
        
        # Two multi-head Transformer convolutional layers
        self.conv1 = TransformerConv(num_node_features, hidden_dim, heads=heads, dropout=0.2)
        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.2)
        
        # Dense layers - fix architecture
        self.dense1 = torch.nn.Linear(hidden_dim * heads, 128)
        self.dense2 = torch.nn.Linear(128, 1)
        
        # Dropout layers
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
    
    def forward(self, x, edge_index, batch, global_attr=None):
        # First Transformer conv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second Transformer conv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Global pooling to get graph-level representation
        # Use PyG's built-in global_mean_pool for reliability
        graph_repr = global_mean_pool(x, batch)
        
        # Dense layers
        out = self.dense1(graph_repr)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.dense2(out)
        
        # Ensure output is 1D for batch processing
        return out.squeeze(-1)

def circuit_data_to_pyg_data(circuit_data: dict) -> Data:
    """
    Convert circuit data to PyTorch Geometric Data object.
    """
    dag = circuit_data['dag']
    label = circuit_data['label']
    global_features = circuit_data['metadata']['global_features']
    
    # Extract node features
    node_features = []
    node_mapping = {}
    
    for i, (node_id, features) in enumerate(dag.nodes(data=True)):
        feature_vector = [
            features['gate_type'],
            features['num_params'],
            features['num_qubits'],
            int(features['is_parametric']),
            int(features['is_two_qubit'])
        ]
        node_features.append(feature_vector)
        node_mapping[node_id] = i
    
    # Extract edge indices
    edge_index = []
    for edge in dag.edges():
        source_idx = node_mapping[edge[0]]
        target_idx = node_mapping[edge[1]]
        edge_index.append([source_idx, target_idx])
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    y = torch.tensor([label], dtype=torch.float)
    global_attr = torch.tensor(global_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y, global_attr=global_attr)

def create_dataloader(dataset: List[dict], batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Create a PyTorch Geometric DataLoader from the circuit dataset.
    Fixed to avoid tensor mismatch issues.
    """
    pyg_data_list = []
    
    for circuit_data in dataset:
        pyg_data = circuit_data_to_pyg_data(circuit_data)
        pyg_data_list.append(pyg_data)
    
    # Use drop_last=True to ensure consistent batch sizes and avoid tensor mismatch
    return DataLoader(pyg_data_list, batch_size=batch_size, shuffle=shuffle, drop_last=True)

def load_and_combine_circuits_from_directory(directory_path: str, max_circuits_per_file: int = None, qubit_range: tuple = None):
    """
    Load and combine circuits from all pickle files in a directory.
    Args:
        directory_path: Path to directory containing pickle files
        max_circuits_per_file: Maximum circuits to load per file
        qubit_range: Tuple (min_qubits, max_qubits) to filter files by qubit count
    """
    print(f"🔄 Loading circuits from {directory_path}...")
    
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(directory_path, "*.pkl"))
    
    # Filter files by qubit range if specified (for dataset_random)
    if qubit_range and "dataset_random" in directory_path:
        min_qubits, max_qubits = qubit_range
        filtered_files = []
        for file in pickle_files:
            filename = os.path.basename(file)
            # Extract qubit count from filename like "basis_rotations+cx_qubits_3_gates_0-19.pkl"
            if "_qubits_" in filename:
                try:
                    qubit_part = filename.split("_qubits_")[1].split("_")[0]
                    num_qubits = int(qubit_part)
                    if min_qubits <= num_qubits <= max_qubits:
                        filtered_files.append(file)
                except:
                    # If parsing fails, include the file
                    filtered_files.append(file)
            else:
                # If no qubit info in filename, include the file
                filtered_files.append(file)
        pickle_files = filtered_files
        print(f"Filtered to {len(pickle_files)} files with {min_qubits}-{max_qubits} qubits")
    
    pickle_files.sort()
    
    print(f"Found {len(pickle_files)} pickle files")
    
    combined_circuits = []
    file_stats = []
    
    for i, pkl_file in enumerate(pickle_files):
        print(f"Loading {os.path.basename(pkl_file)} ({i+1}/{len(pickle_files)})")
        
        try:
            # Load raw circuits from pickle file
            with open(pkl_file, 'rb') as f:
                raw_circuits = pickle.load(f)
            
            # Limit circuits per file if specified
            if max_circuits_per_file:
                raw_circuits = raw_circuits[:max_circuits_per_file]
            
            # Process circuits using QuantumCircuitDAGDataset
            dataset_creator = QuantumCircuitDAGDataset(circuit_file=pkl_file)
            processed_circuits = dataset_creator.create_dataset(num_circuits=len(raw_circuits))
            
            # Add source file info
            for circuit in processed_circuits:
                circuit['source_file'] = os.path.basename(pkl_file)
                circuit['dataset_type'] = os.path.basename(directory_path)
            
            combined_circuits.extend(processed_circuits)
            
            file_stats.append({
                'filename': os.path.basename(pkl_file),
                'num_circuits': len(processed_circuits),
                'num_qubits': processed_circuits[0]['num_qubits'] if processed_circuits else 0
            })
            
            print(f"  Added {len(processed_circuits)} circuits")
            
        except Exception as e:
            print(f"❌ Error loading {pkl_file}: {e}")
            continue
    
    print(f"\n✅ Combined dataset from {directory_path}:")
    print(f"  Total circuits: {len(combined_circuits)}")
    print(f"  Files processed: {len(file_stats)}")
    
    # Print statistics by qubit count
    qubit_stats = defaultdict(int)
    for circuit in combined_circuits:
        qubit_stats[circuit['num_qubits']] += 1
    
    print(f"\n📊 Circuits by qubit count:")
    for qubits in sorted(qubit_stats.keys()):
        print(f"  {qubits} qubits: {qubit_stats[qubits]} circuits")
    
    return combined_circuits, file_stats

def train_and_evaluate_dataset(dataset, dataset_name: str, model_config: dict):
    """
    Train and evaluate model on a dataset.
    """
    print(f"\n🚀 Training on {dataset_name} Dataset")
    print("=" * 70)
    
    print(f"Dataset size: {len(dataset)} circuits")
    
    # Analyze dataset composition
    qubit_counts = defaultdict(int)
    labels = []
    
    for circuit in dataset:
        qubit_counts[circuit['num_qubits']] += 1
        labels.append(circuit['label'])
    
    labels = np.array(labels)
    
    print(f"\n📊 Dataset Composition:")
    for qubits in sorted(qubit_counts.keys()):
        print(f"  {qubits} qubits: {qubit_counts[qubits]} circuits ({100*qubit_counts[qubits]/len(dataset):.1f}%)")
    
    print(f"\n📈 Label Statistics:")
    print(f"  Range: [{labels.min():.4f}, {labels.max():.4f}]")
    print(f"  Mean: {labels.mean():.4f} ± {labels.std():.4f}")
    
    # Split dataset into train/test (80/20)
    train_size = int(0.8 * len(dataset))
    
    # Shuffle dataset for random split
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    print(f"\n🔄 Dataset Split:")
    print(f"  Training set: {len(train_dataset)} circuits (80%)")
    print(f"  Test set: {len(test_dataset)} circuits (20%)")
    
    # Create data loaders
    batch_size = model_config['batch_size']
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuantumCircuitGNN(
        num_node_features=5, 
        heads=model_config['heads'], 
        hidden_dim=model_config['hidden_dim']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    criterion = torch.nn.MSELoss()
    
    print(f"\n🏗️  Model Configuration:")
    print(f"  Device: {device}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hidden dimension: {model_config['hidden_dim']}")
    print(f"  Attention heads: {model_config['heads']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {model_config['learning_rate']}")
    
    # Training
    num_epochs = model_config['epochs']
    model.train()
    
    print(f"\n🏃 Training for {num_epochs} epochs...")
    start_time = time.time()
    
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            try:
                predictions = model(batch.x, batch.edge_index, batch.batch, batch.global_attr)
                
                # Ensure predictions and targets have same shape
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                
                # Additional safety check for tensor sizes
                if predictions.size(0) != batch.y.size(0):
                    print(f"Warning: Prediction size {predictions.size(0)} != Target size {batch.y.size(0)}, skipping batch")
                    continue
                
                loss = criterion(predictions, batch.y)
                
            except RuntimeError as e:
                print(f"Warning: Batch processing error: {e}, skipping batch")
                continue
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            train_losses.append(avg_loss)
        else:
            print(f"Warning: All batches skipped in epoch {epoch + 1}")
            avg_loss = 0.0
        
        # Report progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    
    # Evaluation
    print(f"\n🧪 Evaluating model...")
    model.eval()
    test_predictions = []
    test_labels = []
    test_loss = 0
    num_test_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            try:
                predictions = model(batch.x, batch.edge_index, batch.batch, batch.global_attr)
                
                # Ensure predictions and targets have same shape
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                
                # Additional safety check for tensor sizes
                if predictions.size(0) != batch.y.size(0):
                    print(f"Warning: Test prediction size {predictions.size(0)} != Target size {batch.y.size(0)}, skipping batch")
                    continue
                
                loss = criterion(predictions, batch.y)
                
                test_predictions.extend(predictions.cpu().numpy().flatten())
                test_labels.extend(batch.y.cpu().numpy().flatten())
                test_loss += loss.item()
                num_test_batches += 1
                
            except RuntimeError as e:
                print(f"Warning: Test batch processing error: {e}, skipping batch")
                continue
    
    if num_test_batches > 0:
        avg_test_loss = test_loss / num_test_batches
        test_predictions = np.array(test_predictions)
        test_labels = np.array(test_labels)
    else:
        print("❌ Warning: All test batches were skipped!")
        avg_test_loss = 0.0
        test_predictions = np.array([])
        test_labels = np.array([])
    
    # Calculate metrics
    if len(test_predictions) > 0:
        mae = np.mean(np.abs(test_predictions - test_labels))
        mse = np.mean((test_predictions - test_labels) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate R²
        ss_tot = np.sum((test_labels - np.mean(test_labels)) ** 2)
        ss_res = np.sum((test_labels - test_predictions) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        mae = mse = rmse = r2 = 0.0
    
    print(f"\n{'='*70}")
    print(f"🎯 FINAL RESULTS for {dataset_name}")
    print(f"{'='*70}")
    print(f"Training time: {training_time:.2f}s ({training_time/60:.1f} minutes)")
    print(f"Total circuits: {len(dataset)}")
    print(f"Training circuits: {len(train_dataset)}")
    print(f"Test circuits: {len(test_dataset)}")
    print(f"\n📊 Performance Metrics:")
    print(f"  Test Loss (MSE): {avg_test_loss:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  Root Mean Squared Error: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Performance grade
    if r2 > 0.8:
        grade = "🔥 Excellent"
    elif r2 > 0.6:
        grade = "🎯 Very Good"
    elif r2 > 0.4:
        grade = "✅ Good"
    elif r2 > 0.2:
        grade = "⚠️  Fair"
    else:
        grade = "❌ Poor"
    
    print(f"\n📈 Performance Grade: {grade}")
    
    print(f"\n📈 Label Analysis:")
    print(f"  True labels - Mean: {np.mean(test_labels):.4f} ± {np.std(test_labels):.4f}")
    print(f"  Predictions - Mean: {np.mean(test_predictions):.4f} ± {np.std(test_predictions):.4f}")
    print(f"  Label range: [{min(test_labels):.4f}, {max(test_labels):.4f}]")
    print(f"  Prediction range: [{min(test_predictions):.4f}, {max(test_predictions):.4f}]")
    
    # Show some example predictions
    print(f"\n🔍 Example Predictions:")
    for i in range(min(10, len(test_predictions))):
        error = abs(test_labels[i] - test_predictions[i])
        print(f"  Circuit {i+1}: True={test_labels[i]:.4f}, Predicted={test_predictions[i]:.4f}, Error={error:.4f}")
    
    # Return results
    results = {
        'dataset_name': dataset_name,
        'total_circuits': len(dataset),
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'training_time': training_time,
        'test_loss': avg_test_loss,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'label_mean': np.mean(test_labels),
        'label_std': np.std(test_labels),
        'pred_mean': np.mean(test_predictions),
        'pred_std': np.std(test_predictions),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'hidden_dim': model_config['hidden_dim'],
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': model_config['learning_rate']
    }
    
    return results, model

def main():
    """
    Main function to train on both datasets.
    """
    print("🚀 TRAINING QUANTUM CIRCUIT GNN ON TWO BIG DATASETS")
    print("=" * 80)
    
    # Model configuration
    model_config = {
        'hidden_dim': 128,
        'heads': 4,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 150
    }
    
    all_results = []
    
    # Dataset 1: dataset_tim (Ising circuits)
    print("\n" + "="*80)
    print("📂 DATASET 1: ISING MODEL CIRCUITS (dataset_tim)")
    print("="*80)
    
    # Check if combined dataset already exists
    tim_dataset_file = "quantum_dag_dataset/combined_tim_circuits.pkl"
    if os.path.exists(tim_dataset_file):
        print("Loading existing dataset_tim combined dataset...")
        dataset_creator = QuantumCircuitDAGDataset()
        tim_dataset = dataset_creator.load_dataset("combined_tim_circuits.pkl")
    else:
        print("Creating combined dataset from dataset_tim...")
        tim_circuits, _ = load_and_combine_circuits_from_directory("data/dataset_tim")
        dataset_creator = QuantumCircuitDAGDataset()
        dataset_creator.save_dataset(tim_circuits, "combined_tim_circuits.pkl")
        tim_dataset = tim_circuits
    
    # Train on dataset_tim
    tim_results, tim_model = train_and_evaluate_dataset(tim_dataset, "Ising (dataset_tim)", model_config)
    all_results.append(tim_results)
    
    # Dataset 2: dataset_random (Random circuits)
    print("\n" + "="*80)
    print("📂 DATASET 2: RANDOM BASIS ROTATION CIRCUITS (dataset_random)")
    print("="*80)
    
    # Check if combined dataset already exists
    random_dataset_file = "quantum_dag_dataset/combined_random_circuits_2-6q.pkl"
    if os.path.exists(random_dataset_file):
        print("Loading existing dataset_random combined dataset (2-6 qubits)...")
        dataset_creator = QuantumCircuitDAGDataset()
        random_dataset = dataset_creator.load_dataset("combined_random_circuits_2-6q.pkl")
    else:
        print("Creating combined dataset from dataset_random (2-6 qubits only)...")
        random_circuits, _ = load_and_combine_circuits_from_directory("data/dataset_random", qubit_range=(2, 6))
        dataset_creator = QuantumCircuitDAGDataset()
        dataset_creator.save_dataset(random_circuits, "combined_random_circuits_2-6q.pkl")
        random_dataset = random_circuits
    
    # Train on dataset_random
    random_results, random_model = train_and_evaluate_dataset(random_dataset, "Random (dataset_random)", model_config)
    all_results.append(random_results)
    
    # Save comprehensive results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('two_big_datasets_results.csv', index=False)
    
    # Final comparison
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON OF BOTH DATASETS")
    print("="*80)
    
    print(f"\n🔬 Dataset Comparison:")
    for result in all_results:
        print(f"\n📈 {result['dataset_name']} Dataset:")
        print(f"  Total circuits: {result['total_circuits']:,}")
        print(f"  Training time: {result['training_time']/60:.1f} minutes")
        print(f"  R² Score: {result['r2']:.4f}")
        print(f"  MAE: {result['mae']:.4f}")
        print(f"  RMSE: {result['rmse']:.4f}")
    
    print(f"\n💾 Results saved to: two_big_datasets_results.csv")
    print(f"\n✅ Training completed successfully on both datasets!")

if __name__ == "__main__":
    main() 