import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader

# Load the CORA dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# Define a GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create a DataLoader
loader = DataLoader(data, batch_size=64, shuffle=True)

# Initialize the model and optimizer
model = GraphSAGE(in_channels=dataset.num_features, hidden_channels=128, out_channels=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

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
