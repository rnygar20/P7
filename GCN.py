from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

class GCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16, size_gnn=2):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(num_features, hidden_channels))

        for i in range(size_gnn):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # add dropout layer
        self.dropout = nn.Dropout(p=0.5)

        self.convs.append(GCNConv(hidden_channels, num_classes))

        self.act = F.leaky_relu

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)

            if conv != self.convs[-1]:
                x = self.act(x)

        return F.softmax(x, dim=1)


def train(model,optimizer,data,epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    #print(f'Epoch: {epoch}, Loss: {loss.item()}')
    

def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)  # Predicted probabilities
    probas = torch.nn.functional.softmax(out, dim=1)  # Convert logits to probabilities
    pred = probas.argmax(dim=1)

    # Only consider test data
    true_labels = data.y[data.test_mask].cpu().numpy()
    predicted_labels = pred[data.test_mask].cpu().numpy()

    # AUC for each class
    auc_list = []
    for i in range(probas.size(1)):
        auc_list.append(roc_auc_score((true_labels == i).astype(int), probas[data.test_mask, i].cpu().detach().numpy()))
    avg_auc = sum(auc_list) / len(auc_list)

    # Recall
    recall = recall_score(true_labels, predicted_labels, average='macro')  # Using macro average

    print(f'Test Average AUC: {avg_auc:.4f}')
    print(f'Test Recall: {recall:.4f}')

    # Acuuracy
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    print(f'Test Accuracy: {acc * 100:.2f}%')


# GCN
def run_GCN(dataset, batch_size=1):
    data = dataset[0]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Initialize the model and optimizer
    model = GCN(num_features=data.num_features, num_classes=dataset.num_classes, hidden_channels=100,size_gnn=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # Training loop
    for epoch in range(1, 100):
        train(model,optimizer,data,epoch)
    
    test(model,data)