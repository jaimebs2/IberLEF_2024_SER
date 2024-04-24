import torch
import sys
from src.data_module import EmoSPeechDataModule, EmoSPeechDataset, collate_fn_predict
from audio_language_model.alm import WhisperGemmaClassifier
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import json

def predict(config, test=False):
    torch.set_float32_matmul_precision('high')
    gpu = "cuda:0"
    state = torch.load(config['ckpt'])['state_dict']
    if test:
        test_df = pd.read_csv('data/EmoSPeech_phase_2_test_public.csv')
        test_df["path"] = test_df["id"].apply(lambda x: f"test_segments/{x}")
        dataset = EmoSPeechDataset(test_df, **config)
        data_loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=True,
                collate_fn=collate_fn_predict,
            )
    else:
        db = EmoSPeechDataModule(**config)
        db.setup()
        data_loader = db.val_dataloader()
    
    model = WhisperGemmaClassifier(**config)
    model.configure_model()
    model.load_state_dict(state)
    model.eval()
    model = model.to(device=gpu, dtype=torch.float16)
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if test:
                audio, text_tokens = batch
            else:
                audio, text_tokens, y = batch
                labels = labels + y.tolist()
            audio = audio.to(device=gpu, dtype=torch.float16)
            text_tokens = [text_tokens[0].to(device=gpu, dtype=int)]
            emotions_logits, _ = model(audio.to(device=gpu, dtype=torch.float16), text_tokens)
            pred = torch.argmax(emotions_logits[0].cpu()).tolist()
            preds = preds + [pred]
    
    final_preds = []
    for pred in preds:
        if pred == 0:
            final_preds.append('neutral')
        elif pred == 1:
            final_preds.append('disgust')
        elif pred == 2:
            final_preds.append('anger')
        elif pred == 3:
            final_preds.append('joy')
        elif pred == 4:
            final_preds.append('sadness')
        elif pred == 5:
            final_preds.append('fear')

    if test:
        text_predictions = pd.read_csv('data/text_preds.csv')
        output_df = pd.DataFrame(columns=["id", "task_1", "task_2"])
        output_df["id"] = test_df["id"].str.replace ('.mp3', '', regex = False)
        output_df["task_1"] = text_predictions['0']
        output_df["task_2"] = final_preds
        print (output_df)
        output_df.to_csv ('results.csv', index = False)
        return final_preds, text_predictions
    else:
        final_labels = []
        for label in labels:
            if label == 0:
                final_labels.append('neutral')
            elif label == 1:
                final_labels.append('disgust')
            elif label == 2:
                final_labels.append('anger')
            elif label == 3:
                final_labels.append('joy')
            elif label == 4:
                final_labels.append('sadness')
            elif label == 5:
                final_labels.append('fear')
        return final_preds, final_labels
    
if __name__ == "__main__":
    with open('audio_language_model/config.json') as f:
        config = json.load(f)

    final_preds, final_labels = predict(config, test=False)