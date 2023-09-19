from gnn import GNN
import torch

def main():
    GNN1 = GNN(16, 1)
    print(GNN1(torch.randn(1,1,16)))
    print(torch.randn(1,1,16).shape)
    print(torch.randn(1,1,16))




if __name__ == '__main__':
    main()