import torch
from download_dataset import download_datasets, get_dataset
from GCN import run_GCN
from GraphSAGE import run_GraphSAGE
from GAT import run_GAT

def main():
    #download_datasets()
    cora_dataset = get_dataset('Cora')
    #PubMed_dataset = get_dataset('PubMed')
    #CiteSeer_dataset = get_dataset('CiteSeer')

    #for i in range(10):
     #  run_GCN(CiteSeer_dataset)

    #for i in range(10):
     #   run_GraphSAGE(CiteSeer_dataset)

    for i in range(1):
        run_GAT(cora_dataset)
    



if __name__ == '__main__':
    main()