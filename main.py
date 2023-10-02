import torch
from download_dataset import download_datasets, get_dataset
from GCN import run_GCN
from GraphSAGE import run_GraphSAGE
from GAT import run_GAT

def main():
    download_datasets()
    cora_dataset = get_dataset('Cora')
    CiteSeer_dataset = get_dataset('CiteSeer')
    PubMed_dataset = get_dataset('PubMed')
    #run_GCN(cora_dataset)
    #run_GCN(CiteSeer_dataset)
    #run_GCN(PubMed_dataset)
    #run_GraphSAGE(cora_dataset)
    #run_GraphSAGE(CiteSeer_dataset)
    #run_GraphSAGE(PubMed_dataset)
    #run_GAT(cora_dataset)
    #run_GAT(CiteSeer_dataset)
    #run_GAT(PubMed_dataset)
    



if __name__ == '__main__':
    main()