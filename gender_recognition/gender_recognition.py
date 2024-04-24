from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from typing import List, Dict
from tqdm import tqdm
import librosa
from src.data_module import OdysseyDataModule

model_name= "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
audio_paths = []
device = "cpu"

label2id = {
    "female": 0,
    "male": 1
}

id2label = {
    0: "female",
    1: "male"
}

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(
    pretrained_model_name_or_path=model_name,
    num_labels=2,
)

db = OdysseyDataModule()
audio_path = 'data/train_segments/0a3ad053-f9140267.mp3'
y, sr = librosa.load(audio_path, sr=16000)
audio = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
logits = model(input_values=audio['input_values'], attention_mask=audio['attention_mask']).logits
scores = F.softmax(logits, dim=-1)
pred = torch.argmax(scores, dim=1).cpu().detach().numpy().item()

import pandas as pd
data = pd.read_csv('data/EmoSPeech_phase_2_train.csv')
gender_list = []
for i in tqdm(range(len(data))):
    audio_path = 'data/train_segments/' + data.iloc[i,0]
    y, sr = librosa.load(audio_path, sr=16000)
    audio = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
    logits = model(input_values=audio['input_values'], attention_mask=audio['attention_mask']).logits
    scores = F.softmax(logits, dim=-1)
    pred = torch.argmax(scores, dim=1).cpu().detach().numpy().item()
    gender_list.append(pred)

gender_data = pd.DataFrame({'gender': gender_list})
gender_data.to_csv('data/EmoSPeech_phase_2_train_gender.csv', index=False)
