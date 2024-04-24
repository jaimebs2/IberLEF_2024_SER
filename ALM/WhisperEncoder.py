from transformers import AutoFeatureExtractor
from ALM.modeling_whisper import WhisperForAudioClassification, WhisperEncoder
#from transformers import WhisperForAudioClassification
from transformers import WhisperConfig
import torch
import json

class CustomConfig():
    def __init__(self):
        with open('ALM/whisper_config.json') as f:
            config = json.load(f)
        for key in config:
            setattr(self, key, config[key]) 

class AudioEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.model = WhisperForAudioClassification.from_pretrained(
        #    "openai/whisper-large-v3",
        #)
        self.encoder=WhisperEncoder(CustomConfig())
        self.load_state_dict(torch.load("weights/whisper_large_v3.ckpt"))

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        return encoder_outputs
