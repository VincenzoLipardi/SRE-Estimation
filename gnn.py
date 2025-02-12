import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import os
import pickle

class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)  # Output layer for regression

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def load_dag_data(directory, num_qubit):
    dag_data, labels = [], []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl') and "qubits_"+str(num_qubit) in filename:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as file:
                raw_data = pickle.load(file)
            for item in raw_data:
                dag_data.append(item[0])
                labels.append(item[1])

    return dag_data, labels


dag_data, labels = load_dag_data(directory="data/random_circuits_dag", num_qubit=2) 

loader = DataLoader(dag_data, batch_size=32, shuffle=True)

model = GNNModel(num_node_features=2, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
