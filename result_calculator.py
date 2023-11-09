# import file result_GCN_cora.csv and print average of each column


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import results_GAT.csv and calculate the average of each column for each dataset and loss function and save the results to a new file
df = pd.read_csv('results_GAT.csv', header=0)
df = df.groupby(['dataset','lossfunction']).mean()
df = df.iloc[:, :6]
df = df.round(3)
print(df)



"""
df = pd.read_csv('results_GCN_cora.csv', header=None)
df.columns = ['avg_auc', 'recall', 'acc', 'f1']
print("GCN cora")
print(df.mean(axis=0))

# import file result_GCN_citeseer.csv and print average of each column
df = pd.read_csv('results_GCN_cite.csv', header=None)
df.columns = ['avg_auc', 'recall', 'acc', 'f1']
print("GCN citeseer")
print(df.mean(axis=0))

# import file result_GCN_pubmed.csv and print average of each column
df = pd.read_csv('results_GCN_pubmed.csv', header=None)
df.columns = ['avg_auc', 'recall', 'acc', 'f1']
print("GCN pubmed")
print(df.mean(axis=0))

# import file result_GraphSAGE_cora.csv and print average of each column
df = pd.read_csv('results_GraphSAGE_cora.csv', header=None)
df.columns = ['avg_auc', 'recall', 'acc', 'f1']
print("GraphSAGE cora")
print(df.mean(axis=0))

# import file result_GraphSAGE_citeseer.csv and print average of each column
df = pd.read_csv('results_GraphSAGE_cite.csv', header=None)
df.columns = ['avg_auc', 'recall', 'acc', 'f1']
print("GraphSAGE citeseer")
print(df.mean(axis=0))

# import file result_GraphSAGE_pubmed.csv and print average of each column
df = pd.read_csv('results_GraphSAGE_pubmed.csv', header=None)
df.columns = ['avg_auc', 'recall', 'acc', 'f1']
print("GraphSAGE pubmed")
print(df.mean(axis=0))
"""
'''# import file result_GAT_cora.csv and print average of each column
df = pd.read_csv('results_GAT_cora.csv', header=None)
df.columns = ['avg_auc', 'recall', 'acc', 'f1']
print("GAT cora")
print(df.mean(axis=0))

# import file result_GAT_citeseer.csv and print average of each column
df = pd.read_csv('results_GAT_cite.csv', header=None)
df.columns = ['avg_auc', 'recall', 'acc', 'f1']
print("GAT citeseer")
print(df.mean(axis=0))

# import file result_GAT_pubmed.csv and print average of each column
df = pd.read_csv('results_GAT_pubmed.csv', header=None)
df.columns = ['avg_auc', 'recall', 'acc', 'f1']
print("GAT pubmed")
print(df.mean(axis=0))'''




