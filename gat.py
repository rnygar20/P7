import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader

# Load the CORA dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# Define a GAT model
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_heads):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, num_classes, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create a DataLoader
loader = DataLoader(data, batch_size=64, shuffle=True)

# Initialize the model and optimizer
model = GATModel(in_channels=dataset.num_features, hidden_channels=64, num_classes=dataset.num_classes, num_heads=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
for epoch in range(1, 201):  # You can adjust the number of epochs
    loss = train()
    print(f'Epoch: {epoch}, Loss: {loss:.4f}')

# Evaluation function
def evaluate():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    return acc

# Evaluate the model
test_accuracy = evaluate()
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
