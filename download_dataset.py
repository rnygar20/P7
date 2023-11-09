from torch_geometric.datasets import Planetoid
import os
import torch_geometric.transforms as T


def download_datasets():
    dataset_names = ['Cora', 'CiteSeer', 'PubMed']
    data_dir = './dataset/'

    for name in dataset_names:
        dataset_path = os.path.join(data_dir, name)
    
        # Check if the dataset directory already exists
        if os.path.exists(dataset_path):
            print(f"Dataset {name} already exists in {data_dir}.") 
        else:
            dataset = Planetoid(root=data_dir, name=name, transform=T.NormalizeFeatures())
            print(f"Dataset: {name}")
            print(f"Number of nodes: {dataset[0].num_nodes}")
            print(f"Number of edges: {dataset[0].num_edges}")
            print(f"Number of classes: {dataset.num_classes}")
            print(f"Number of features: {dataset.num_features}")
            print(f"Number of graphs: {len(dataset)}")
    

def get_dataset(name):
    """
    Load a dataset from Planetoid and return it.

    Args:
        name (str): Name of the dataset ('Cora', 'CiteSeer', or 'PubMed').

    Returns:
        torch_geometric.data.Dataset: The loaded dataset.
    """
    data_dir = './dataset/'
    dataset = Planetoid(root=data_dir, name=name, split='full')
    return dataset