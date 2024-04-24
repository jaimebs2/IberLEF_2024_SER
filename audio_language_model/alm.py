import torch
import pytorch_lightning as pl
import torchmetrics as tm
from torchmetrics.classification import MulticlassF1Score
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from transformers import WhisperConfig, GemmaConfig, GemmaForCausalLM
from transformers.models.whisper.modeling_whisper import WhisperEncoder
import json
from deepspeed.ops.adam import FusedAdam
from audio_language_model.emotion2vec import Emotion2vec

class WhisperGemmaClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.num_classes = self.kwargs.get("num_classes", 6)
        self.use_lora = self.kwargs.get("use_lora", False)
        self.use_lora_phi = self.kwargs.get("use_lora_phi", False)
        self.lr = self.kwargs.get("lr", 1e-4)
        self.trainable_layers = self.kwargs.get("trainable_layers", ["transformer"])
        self.transcript = self.kwargs.get("transcript", False)
        self.phones = self.kwargs.get("phones", False)
        self.blank_token = self.kwargs.get("blank_token", 198)
        self.batch_size = self.kwargs.get("batch_size", 8)
        self.lr_scheduler = self.kwargs.get("lr_scheduler", None)
        self.lr_decay = self.kwargs.get("lr_decay", 0.9)
        self.incremental_training = self.kwargs.get("incremental_training", False)
        self.optimizer_type = self.kwargs.get("optimizer_type", "adam")
        self.llm_name = self.kwargs.get("llm_name", "phi")
        self.batch_size = self.kwargs.get("batch_size", 4)
        self.balanced = self.kwargs.get("balanced", False)
        self.augmented = self.kwargs.get("augmented", False)
        self.max_epochs = self.kwargs.get("max_epochs", 20)
        self.start_factor = self.kwargs.get("start_factor", 1.0)
        self.end_factor = self.kwargs.get("end_factor", 0.0)
        self.total_iters = self.kwargs.get("total_iters", 10)
        self.deepspeed = self.kwargs.get("deepspeed", False)
        self.predict = self.kwargs.get("predict", False)
        self.use_audio = self.kwargs.get("use_audio_encoder", True)
        self.audio_encoder_name = self.kwargs.get("audio_encoder_name", "whisper-large-v3")
        self.trainable_layers = self.kwargs.get("trainable_layers", ["projector"])
        self.use_gemma_finetuned = self.kwargs.get("use_gemma_finetuned", False)
        self.save_hyperparameters()

        # odyssey: ' neutral', 'disgust', 'anger', 'joy', 'sadness', 'fear'
        self.space_indices = torch.tensor([17120, 41497, 19456, 4915, 9270, 7440])#[' neutral disgust angry happy sad fear']
        class_counts = [1166, 705, 399, 362, 345, 23]

        total_samples = sum(class_counts)
        class_weights = [total_samples / count for count in class_counts]
        weights = torch.tensor(class_weights)
        self.loss_fn = torch.nn.CrossEntropyLoss(weights)
        self.acc = tm.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1 = MulticlassF1Score(num_classes=self.num_classes, average="macro")
        self.train_step_outputs = []
        self.train_step_targets = []
        self.val_step_outputs = []
        self.val_step_targets = []

    def configure_model(self):
        '''
        with open('ALM/whisper_config.json') as f:
            whisper_config = json.load(f)
        self.audio_encoder = WhisperEncoder(WhisperConfig(**whisper_config))
        self.projector = nn.Linear(1280, 2048)
        #with open('ALM/gemma_config.json') as f:
        #    gemma_config = json.load(f)
        #self.llm = GemmaForCausalLM(GemmaConfig(**gemma_config)).to(dtype=torch.float16)
        self.llm = GemmaForCausalLM.from_pretrained("google/gemma-2b").to(dtype=torch.float16)
        '''
        if self.use_gemma_finetuned:
            self.model = torch.load("checkpoints/WGF")
        else:
            self.model = torch.load("weights/main.ckpt")
        if self.audio_encoder_name=="emotion2vec":
            emo2vec = Emotion2vec()
            self.model.audio_encoder = emo2vec
            self.model.projector = nn.Linear(768, 2048)
        if not self.use_audio:
            self.model.llm.lm_head = nn.Linear(self.model.llm.lm_head.in_features, 6)
    
    def _pad_audio(self, x):
        padding_masks = []
        for item in x:
            means=item.mean(0)
            for i in range(1,len(means)):
                if means[-i].item()-means[-i-1].item()!=0:
                    last_position=i
                    mask = torch.ones(x.shape[2]-last_position)
                    padding_masks.append(mask)
                    break
        
        max_len = max([len(mask) for mask in padding_masks])
        padding_masks = [
            torch.nn.functional.pad(mask, (0, max_len - len(mask)))
            for mask in padding_masks
        ]
        padding_mask = torch.stack(padding_masks).to(dtype=x.dtype)
        x = x[:, :, :padding_mask.size(1)]
        return x, padding_mask

    def _pad_transcript(self, prompt):
        for i in range(len(prompt)):
            if len(prompt[i])==1:
                prompt[i]=prompt[i][0]
        max_len = max([len(x) for x in prompt])
        padded_prompt = []
        prompt_attn_mask = []
        for p in prompt:
            split_index = [i for i, num in enumerate(p) if num == 108]
            padded_prompt.append(torch.concat((p[:split_index[3]], torch.zeros(max_len - len(p)).to(device=p.device, dtype=p.dtype), p[split_index[3]:]),dim=0))
            prompt_attn_mask.append(torch.concat((torch.ones(len(p[:split_index[3]])), torch.zeros(max_len - len(p)), torch.ones(len(p[split_index[3]:]))),dim=0))
        text_attn_mask = torch.stack(prompt_attn_mask).to(dtype=torch.int32)
        return padded_prompt, text_attn_mask

    def _combine_features(self, audio_features, text_features, blank_token_position, text_attn_mask):
        combined_feautes = torch.cat([text_features[:,:blank_token_position,:],audio_features,text_features[:,(blank_token_position+1):,:]],dim=1)
        if text_attn_mask is not None:
            text_attn_mask = text_attn_mask.unsqueeze(2).expand(-1,-1, 2048)
            audio_attn_mask = torch.ones_like(audio_features).to(dtype=torch.int32, device=audio_features.device)
            attn_mask = torch.cat([text_attn_mask[:,:blank_token_position,:], audio_attn_mask, text_attn_mask[:,(blank_token_position+1):,:]], dim=1)
        else:
            attn_mask = None
        return combined_feautes, attn_mask

    def forward(self, audio, text_tokens, audio_mask=None):
        if self.use_audio:
            if self.audio_encoder_name!="emotion2vec":
                audio, audio_mask = self._pad_audio(audio)
            if self.audio_encoder_name=="emotion2vec":
                _, encoder_outputs = self.model.audio_encoder(audio, audio_mask)
                audio_features = encoder_outputs['x']
            else:
                encoder_outputs = self.model.audio_encoder(audio)
                audio_features = encoder_outputs[0]
            audio_features = self.model.projector(audio_features)
        else:
            audio_features = torch.zeros(audio.shape[0], 1, 2048).to("cuda")
        if self.batch_size!=1:
            text_tokens, text_attn_mask = self._pad_transcript(text_tokens)
            blank_token_position = (text_tokens[0]==109).nonzero(as_tuple=True)[0][0].item()
            text_features = self.model.llm.model.embed_tokens(torch.stack(text_tokens))
            combined_feautes, attn_mask = self._combine_features(audio_features, text_features, blank_token_position, text_attn_mask.to(device=audio_features.device))
        else:
            text_tokens = text_tokens[0]
            text_attn_mask = None
            blank_token_position = (text_tokens==109).nonzero(as_tuple=True)[0][0].item()
            text_features = self.model.llm.model.embed_tokens(text_tokens)
            text_features=text_features.unsqueeze(0)
            combined_feautes, attn_mask = self._combine_features(audio_features, text_features, blank_token_position, text_attn_mask)
            
        logits = self.model.llm(attention_mask=attn_mask, inputs_embeds=combined_feautes)['logits']
        next_token_logits = logits[:, -1, :]
        emotions_logits = next_token_logits[:, self.space_indices.to(next_token_logits.device)]
        return emotions_logits, next_token_logits

    def training_step(self, batch):
        if self.audio_encoder_name=="emotion2vec":
            audio, audio_mask, text_tokens, y = batch
        else:
            audio, text_tokens, y = batch
            audio_mask = None

        emotions_logits, _ = self(audio, text_tokens, audio_mask)
        loss = self.loss_fn(emotions_logits, y)

        self.log(
            "train/loss", 
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            sync_dist=True
        )
        train_preds = torch.argmax(emotions_logits, dim=1)
        self.train_step_outputs.extend(train_preds)
        self.train_step_targets.extend(y)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(
            {
                "train/acc": self.acc(torch.stack(self.train_step_outputs), torch.stack(self.train_step_targets)), 
                "train/f1": self.f1(torch.stack(self.train_step_outputs), torch.stack(self.train_step_targets))
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        self.acc.reset()
        self.f1.reset()
        self.train_step_outputs.clear()
        self.train_step_targets.clear()

    def validation_step(self, batch, batch_idx):
        if self.audio_encoder_name=="emotion2vec":
            audio, audio_mask, text_tokens, y = batch
        else:
            audio, text_tokens, y = batch
            audio_mask = None

        emotions_logits, _ = self(audio, text_tokens, audio_mask)
        loss = self.loss_fn(emotions_logits, y)
        self.log(
            "val/loss", 
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            sync_dist=True
        )
        val_preds = torch.argmax(emotions_logits, dim=1)
        self.val_step_outputs.extend(val_preds)
        self.val_step_targets.extend(y)

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "val/acc": self.acc(torch.stack(self.val_step_outputs), torch.stack(self.val_step_targets)), 
                "val/f1": self.f1(torch.stack(self.val_step_outputs), torch.stack(self.val_step_targets))
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        self.acc.reset()
        self.f1.reset()
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.lr
                                         )
        if self.optimizer_type == "adamw":
            if self.deepspeed:
                optimizer = FusedAdam(self.parameters(),
                                            lr=self.lr, 
                                            betas=[0.9, 0.98],
                                            eps=1e-5,
                                            weight_decay=0.01
                                            )
            else:
                optimizer = torch.optim.AdamW(self.parameters(), 
                                            lr=self.lr, 
                                            betas=[0.9, 0.98],
                                            eps=1e-5,
                                            weight_decay=0.01
                                            )
        if self.lr_scheduler == "linear":
            scheduler = lr_scheduler.LinearLR(optimizer, 
                                              start_factor=self.start_factor, 
                                              end_factor=self.end_factor, 
                                              total_iters=self.total_iters
                                              )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "name":"lr_scheduler"}]
        return [optimizer]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
