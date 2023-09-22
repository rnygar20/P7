import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

# Load the Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

# Create a DataLoader
loader = DataLoader(data, batch_size=1, shuffle=True)

for batch in loader:
    print(batch)
    break
#print(loader.dataset.x[0].size())
""" import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Apply first GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Apply second GCN layer
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

import torch.optim as optim

# Initialize the model and optimizer
model = GCN(num_features=data.num_features, num_classes=dataset.num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Training loop
for epoch in range(1, 201):
    train(epoch)


def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    print(f'Test Accuracy: {acc * 100:.2f}%')

test() """

'''
import torch

class NN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(NN, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return  self.layers(x).squeeze(2)
'''