from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim

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

        return F.softmax(x, dim=1)


def train(model,optimizer,data,epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')


def test(model,data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    print(f'Test Accuracy: {acc * 100:.2f}%')

# GCN
def run_GCN(dataset, batch_size=1):
    data = dataset[0]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Initialize the model and optimizer
    model = GCN(num_features=data.num_features, num_classes=dataset.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # Training loop
    for epoch in range(1, 100):
        train(model,optimizer,data,epoch)
    
    test(model,data)