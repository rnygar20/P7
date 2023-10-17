import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#import data from results.csv and set columns to avg_auc, recall, acc

data = pd.read_csv('results.csv', names=['avg_auc', 'recall', 'acc'])


sns.set_style(style="whitegrid")
sns.set_context("talk")

# Create a new figure for the Accuracy plot
plt.figure(figsize=(8, 5))
ax = sns.violinplot(y=data['acc'], inner="quartile", palette="pastel")
ax.set(ylabel='')
plt.title("Accuracy")
sns.despine(left=True, bottom=True)
plt.savefig('figures/acc_violin.png', dpi=300)


""" # Create a new figure for the Average AUC plot
plt.figure(figsize=(10, 5))
avg = sns.violinplot(y=data['avg_auc'], inner="quartile", palette="pastel")
avg.set(ylabel='')
plt.title("Average AUC")
sns.despine(left=True, bottom=True)
plt.savefig('figures/avg_auc_violin.png', dpi=300)

# Create a new figure for the Recall plot
plt.figure(figsize=(10, 5))
rec = sns.violinplot(y=data['recall'], inner="quartile", palette="pastel")
rec.set(ylabel='')
plt.title("Recall")
sns.despine(left=True, bottom=True)
plt.savefig('figures/recall_violin.png', dpi=300) """

