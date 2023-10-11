import torch
from download_dataset import download_datasets, get_dataset
from GCN import run_GCN
from GraphSAGE import run_GraphSAGE
from GAT import run_GAT

def main():
    download_datasets()
    cora_dataset = get_dataset('Cora')
    run_GCN(cora_dataset)

    



if __name__ == '__main__':
    main()