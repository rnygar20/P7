
import torch_geometric
from torch_geometric.datasets import CitationFull
datasets = ['Cora', 'CiteSeer', 'PubMed']

for dataset in datasets:
    data = CitationFull(root='./dataset', name=dataset, )

    # print information about the dataset
    print("Name of the dataset: {}".format(dataset))
    print('====================')
    print('Dataset: {}'.format(data))
    print('====================')
    print('Number of graphs: {}'.format(len(data)))
    print('Number of features: {}'.format(data.num_features))
    print('Number of classes: {}'.format(data.num_classes))
    print('Number of edge features: {}'.format(data.num_edge_features))
    print("Number of nodes: {}".format(data[0].num_nodes))
    print("Number of edges: {}".format(data[0].num_edges))
