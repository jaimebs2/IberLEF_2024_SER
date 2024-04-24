from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from audio_language_model.predict import predict
import numpy as np
import wandb
api = wandb.Api()

run = api.run("jbellver/IberLEF_2024/4k060x1l")
run.name = "WG_b4a6_lr3i5_4_MGE"
run.group = "WG_b6a6_lr3i5"
run.update()

with open('audio_language_model/config.json') as f:
    config = json.load(f)
config['predict'] = True
config['batch_size'] = 1

preds, labels = predict(config, test=True)

print(classification_report(labels, preds))


cm = confusion_matrix(labels, preds)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 10))
plt.title("Confusion Matrix")
sns.heatmap(cm, annot=True, fmt=".2%", cmap='Blues', xticklabels=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness'], yticklabels=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness'])
plt.savefig("conf.png")

import pandas as pd
gender = pd.read_csv('data/EmoSPeech_phase_2_train_gender.csv')
all = pd.read_csv('data/EmoSPeech_phase_2_train.csv')
df = pd.concat([all,gender],axis=1)

label_counts_per_gender = df.groupby('gender')['label'].value_counts()
print(label_counts_per_gender)