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