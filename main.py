#from gnn import GNN
import torch
from download_dataset import get_dataset

def main():
    data = get_dataset('Cora')
    print(data.num_node_features)
    print(data.num_classes)
    print(data.num_edge_features)
    print(data[0].num_nodes)
    print(type(data[0]))
    print(data[0].edge_index)

    node_features = data.x

    print("Node features shape:", node_features.shape)
    print("Node features for the first node:", node_features[0])




if __name__ == '__main__':
    main()