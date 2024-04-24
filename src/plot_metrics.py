from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.data_module import EmoSPeechDataModule

db = EmoSPeechDataModule()
db.setup()
test_labels = [batch[2].item() for batch in db.val_dataloader()]
preds = pd.read_csv("whisperphi_zeroshot_test.csv")['0'].tolist()
# For WhisperGemma zeroshot:
##['N', 'S', 'D', 'H', 'F', 'A', 'C', 'U']
##neutral, dis, ang, joy, sad, f
'''
for i in range(len(preds)):
    if preds[i]==1:
        preds[i]=4
    elif preds[i]==2:
        preds[i]=1
    elif preds[i]==4:
        preds[i]=5
    elif preds[i]==5:
        preds[i]=2
    elif preds[i]==6:
        preds[i]=1
    elif preds[i]==7:
        preds[i]=3
'''

f1_macro=f1_score(test_labels, preds, average='macro')
acc = accuracy_score(test_labels, preds)
valid_conf_matrix = confusion_matrix(test_labels, preds)

# Table
df = pd.DataFrame({'Accuracy': round(acc,4), 'F1 Macro': round(f1_macro,4)}, index=['Test'])
plt.figure(figsize=(6, 2)) 
plt.title('Whisper-large-V3 + Gemma 2b Zeroshot Metrics')
plt.table(cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center',
        colWidths=[0.2]*len(df.columns),
        cellLoc='center',
        colColours=['lightblue']*3,
        rowColours=['lightblue']*3,
        cellColours=None,
        rowLoc='center'
        )
plt.axis('off') 
plt.savefig('whispergemma_zeroshot_metrics.png', bbox_inches='tight', pad_inches=0.1)
plt.close()

# Confusion Matrix
per_conf_matrix = []
for i in range(len(valid_conf_matrix)):
    new_values = list(valid_conf_matrix[i]/valid_conf_matrix.sum(axis=1)[i])
    new_values = [round(value, 2) for value in new_values]
    per_conf_matrix.append(new_values)
per_conf_matrix = np.array(per_conf_matrix)

plt.figure(figsize=(10, 10))
sns.heatmap(per_conf_matrix, annot=True, cmap='Blues', xticklabels=['neutral', 'disgust', 'anger', 'joy', 'sadness', 'fear'], yticklabels=['neutral', 'disgust', 'anger', 'joy', 'sadness', 'fear'])
plt.xlabel('Predicted', fontsize=15)
plt.ylabel('True', fontsize=15)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(rotation=0,fontsize=12)
plt.title('Whisper-large-v3 + Gemma 2b Zeroshot Validation Confusion Matrix', fontsize=20)
plt.savefig('whispergemma_zeroshot_conf_matrix.png')
plt.close()
