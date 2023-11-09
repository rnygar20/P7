import torch
from download_dataset import download_datasets, get_dataset
from GCN import run_GCN
from GraphSAGE import run_GraphSAGE
from GAT import run_GAT

def run_model(modeltype,dataset,dataset_name,lossfunction_name):
    if modeltype == 'GCN':
        run_GCN(dataset,dataset_name,lossfunction_name)
    elif modeltype == 'GraphSAGE':
        run_GraphSAGE(dataset,dataset_name,lossfunction_name)
    elif modeltype == 'GAT':
        run_GAT(dataset,dataset_name,lossfunction_name)
    else:
        print('Invalid model type')

def main():
    #download_datasets()
    Cora_dataset = get_dataset('Cora')
    PubMed_dataset = get_dataset('PubMed')
    CiteSeer_dataset = get_dataset('CiteSeer')

    # Run 10 iterations of GAT for each dataset and loss function
    for i in range(10):
       run_model("GAT",Cora_dataset, "Cora", "CEL")

    for i in range(10):
       run_model("GAT",Cora_dataset, "Cora", "WCEL")

    for i in range(10):
       run_model("GAT",PubMed_dataset, "PubMed", "CEL")

    for i in range(10):
       run_model("GAT",PubMed_dataset, "PubMed", "WCEL")

    for i in range(10):
       run_model("GAT",CiteSeer_dataset, "CiteSeer", "CEL")

    for i in range(10):
       run_model("GAT",CiteSeer_dataset, "CiteSeer", "WCEL")

    # Run 10 iterations of GraphSAGE for each dataset and loss function
    for i in range(10):
       run_model("GraphSAGE",Cora_dataset, "Cora", "CEL")

    for i in range(10):
       run_model("GraphSAGE",Cora_dataset, "Cora", "WCEL")

    for i in range(10):
       run_model("GraphSAGE",PubMed_dataset, "PubMed", "CEL")

    for i in range(10):
       run_model("GraphSAGE",PubMed_dataset, "PubMed", "WCEL")

    for i in range(10):
       run_model("GraphSAGE",CiteSeer_dataset, "CiteSeer", "CEL")

    for i in range(10):
       run_model("GraphSAGE",CiteSeer_dataset, "CiteSeer", "WCEL")

    # Run 10 iterations of GCN for each dataset and loss function
    for i in range(10):
       run_model("GCN",Cora_dataset, "Cora", "CEL")

    for i in range(10):
       run_model("GCN",Cora_dataset, "Cora", "WCEL")

    for i in range(10):
       run_model("GCN",PubMed_dataset, "PubMed", "CEL")

    for i in range(10):
       run_model("GCN",PubMed_dataset, "PubMed", "WCEL")

    for i in range(10):
       run_model("GCN",CiteSeer_dataset, "CiteSeer", "CEL")

    for i in range(10):
       run_model("GCN",CiteSeer_dataset, "CiteSeer", "WCEL")
    


if __name__ == '__main__':
    main()