import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from torcheval.metrics import MulticlassAUROC
import time
import pandas as pd
import os


class GAT(nn.Module):
    """ def __init__(self, num_features, num_classes, hidden_channels=16, size_gnn=2, dropout=0.5, num_heads=8):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()

        self.convs.append(GATConv(num_features, hidden_channels, heads=num_heads))

        for i in range(size_gnn):
            self.convs.append(GATConv(hidden_channels*num_heads, hidden_channels, heads=num_heads))
        
        # add dropout layer
        self.dropout = nn.Dropout(p=dropout)

        self.convs.append(GATConv(hidden_channels*num_heads, num_classes, heads=num_heads))

        self.act = F.leaky_relu """
    
    def __init__(self, num_features, num_classes, hidden_channels=16, size_gnn=2, dropout=0.5, num_heads=8):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()

        self.convs.append(GATConv(num_features, hidden_channels, heads=num_heads, dropout=dropout))
        
        # add dropout layer
        #self.dropout = nn.Dropout(p=dropout)

        self.convs.append(GATConv(hidden_channels*num_heads, num_classes))

        self.act = F.leaky_relu

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)

            if conv != self.convs[-1]:
                x = self.act(x)

        return F.softmax(x, dim=1)


def train(model,optimizer,data,epoch,lossfunction):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = lossfunction(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    #print(f'Epoch: {epoch}, Loss: {loss.item()}')
    

def test(model, data, training_time_list, dataset_name, lossfunction_name):
    model.eval()
    out = model(data.x, data.edge_index)  # Predicted probabilities
    probas = torch.nn.functional.softmax(out, dim=1)  # Convert logits to probabilities
    probas_test_set = probas[data.test_mask]
    pred = probas.argmax(dim=1)

    # Only consider test data
    true_labels = data.y[data.test_mask]
    predicted_labels = pred[data.test_mask]

    # Recall
    recall = recall_score(true_labels, predicted_labels, average='macro')  # Using macro average
    print(f'Test Recall: {recall:.4f}')

    # Acuuracy
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    print(f'Test Accuracy: {acc * 100:.2f}%')

    # F1 score macro
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    print(f'Test F1-macro score: {f1_macro:.4f}')

    # F1 score micro
    f1_micro = f1_score(true_labels, predicted_labels, average='micro')
    print(f'Test F1-micro score: {f1_micro:.4f}')

    # AUC-ROC from torcheval
    metric = MulticlassAUROC(num_classes=probas_test_set.size(1), average="macro")
    metric.update(probas_test_set, true_labels)
    auroc = metric.compute()
    print(f'Test AUROC: {auroc:.4f}')

    # Calculate average training time
    avg_training_time = sum(training_time_list) / len(training_time_list)
    print(f'Average training time: {avg_training_time:.4f}')

    # Calculate accuracy for each class
    class_correct = list(0. for i in range(probas_test_set.size(1)))
    class_total = list(0. for i in range(probas_test_set.size(1)))
    with torch.no_grad():
        for i in range(len(true_labels)):
            label = true_labels[i]
            class_correct[label] += predicted_labels[i].eq(label).item()
            class_total[label] += 1
    
    # Print accuracy for each class
    for i in range(probas_test_set.size(1)):
        print(f'Accuracy of {i}: {100 * class_correct[i] / class_total[i]}%')

    # Calculate F1 score for each class
    f1_list = []
    for i in range(probas_test_set.size(1)):
        f1_list.append(f1_score(true_labels == i, predicted_labels == i))
    
    # Print F1 score for each class
    for i in range(probas_test_set.size(1)):
        print(f'F1 score of {i}: {f1_list[i]:.4f}')


    # AUC for each class
    auc_list = []
    for i in range(probas.size(1)):
        auc_list.append(roc_auc_score((true_labels == i).type(torch.int), probas[data.test_mask, i].cpu().detach().numpy()))
    
    # Print AUC for each class
    for i in range(probas.size(1)):
        print(f'AUC of {i}: {auc_list[i]:.4f}')

    # Create dataframe and include the columns: recall, acc, f1_macro, f1_micro, auroc, avg_training_time and the accuracy for each class and the AUC for each class

    df_dict = {'dataset': dataset_name, 'lossfunction': lossfunction_name, 'recall': [recall], 'acc': [acc], 'f1_macro': [f1_macro], 'f1_micro': [f1_micro], 'auroc': [auroc.item()], 
               'avg_training_time': [avg_training_time]}
    
    for i in range(probas.size(1)):
        df_dict[f'acc_class_{i}'] = [100 * class_correct[i] / class_total[i]]
        df_dict[f'f1_class_{i}'] = [f1_list[i]]
        df_dict[f'auc_class_{i}'] = [auc_list[i]]

    df = pd.DataFrame(df_dict)

    # Write df to csv file
    if os.path.exists('results_GAT.csv'):
        df.to_csv('results_GAT.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('results_GAT.csv', mode='a', header=True, index=False)


# GAT
def run_GAT(dataset,dataset_name,lossfunction_name):
    batch_size = 256
    hidden_channels = 100
    size_gnn = 1
    lr = 0.01
    weight_decay = 5e-4
    epochs = 100
    dropout = 0.5
    num_heads = 8
    training_time_list = []

    data = dataset[0]
    if lossfunction_name == "WCEL":
        label_counts = torch.bincount(data.y)
        label_weights = 1.0 / label_counts.to(torch.float32)
        label_weights = label_weights / label_weights.sum()
        lossfunction = nn.CrossEntropyLoss(weight=label_weights)
    elif lossfunction_name == "CEL":
        lossfunction = nn.CrossEntropyLoss()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Initialize the model and optimizer
    model = GAT(num_features=data.num_features, num_classes=dataset.num_classes, hidden_channels=hidden_channels,
                size_gnn=size_gnn, dropout=dropout, num_heads=num_heads)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Training loop
    for epoch in range(1, epochs+1):
        for data in loader:
            time_start = time.time()
            train(model,optimizer,data,epoch,lossfunction)
            training_time_list.append(time.time() - time_start)
    
    test(model,data, training_time_list, dataset_name, lossfunction_name)


def run_GAT_hyp(dataset,lr,hidden_channels,size_gnn,weight_decay,epochs,dropout,batch_size):
    data = dataset[0]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Initialize the model and optimizer
    model = GAT(num_features=data.num_features, num_classes=dataset.num_classes, hidden_channels=hidden_channels,
                size_gnn=size_gnn, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Training loop
    for epoch in range(1, epochs+1):
        for data in loader:
            train(model,optimizer,data,epoch)
    
    #test(model,data)

    #get validation loss
    model.eval()
    out = model(data.x, data.edge_index)
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    return val_loss

