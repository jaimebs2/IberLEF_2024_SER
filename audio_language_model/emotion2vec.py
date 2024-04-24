from funasr import AutoModel
from funasr.utils.load_utils import load_audio_text_image_video
import torch.nn.functional as F
import numpy as np
import torch
from src.data_module import EmoSPeechDataModule
import json

tokenizer = torch.load("weights/emotion2vec_tokenizer.ckpt")
source = load_audio_text_image_video('data/train_segments/0a3b8e02-d5989d15.mp3', fs=16000, audio_fs=16000,
                                                        data_type="sound", tokenizer=tokenizer)

class Emotion2vec(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = torch.load("weights/emotion2vec.ckpt")
        #['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy', '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>']
        #my model: ['neutral', 'disgust', 'anger', 'joy', 'sadness', 'fear']

    def forward(self, input, mask):
        source=F.layer_norm(input, input.shape)
        source = source.view(source.shape[0], -1)
        feats = self.model.extract_features(source, padding_mask=mask)
        x = feats['x']
        scores = []

        x = x.mean(dim=1)
        output = self.model.proj(x)
        scores = torch.softmax(output, dim=-1)
        return scores, feats

if __name__=="__main__":
    def translate_labels(self, values):
        predictions = []
        for p in values:
            if p==0:
                predictions.append(2)
            if p==1:
                predictions.append(1)
            if p==2:
                predictions.append(5)
            if p==3 or p==7:
                predictions.append(3)
            if p==4 or p==5 or p==8:
                predictions.append(0)
            if p==6:
                predictions.append(4)
        return predictions

    with open('audio_language_model/config.json') as f:
        config = json.load(f)
    db = EmoSPeechDataModule(**config)
    db.setup()
    train = db.train_dataloader()
    tokenizer = torch.load("weights/emotion2vec_tokenizer.ckpt")
    model = Emotion2vec()
    model.to("cuda")
    predictions = []
    labels = []
    from tqdm import tqdm
    for batch in tqdm(train):
        audio_path, _, label = batch
        labels = labels + label.tolist()
        sources = []
        for audio in audio_path:
            source = load_audio_text_image_video(audio, fs=16000, audio_fs=16000,
                                                            data_type="sound", tokenizer=tokenizer)
            sources.append(source)
        input, mask = model.pad(sources)
        scores, feats = model(input.to("cuda"), mask.to("cuda"))
        output = torch.argmax(scores, dim=1)
        predictions = predictions + output.tolist()

    preds = translate_labels(predictions, predictions)
    from sklearn.metrics import f1_score
    f1_score(labels, preds, average='macro') #0.21596651335101344