from ALM.WhisperEncoder import AudioEncoder
from ALM.LLM_original import LLM #_original
import torch
import pytorch_lightning as pl
import torchmetrics as tm
from torchmetrics.classification import MulticlassF1Score
import configparser
from deepspeed.ops.adam import FusedAdam
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

main_config = configparser.ConfigParser()
main_config.read('config.ini')

class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value =torch. nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):

        # Calculate query, key, and value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(
            torch.tensor(self.hidden_dim).float()
        )

        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Apply attention weights to the value
        attended_values = torch.matmul(attention_weights, value)

        return attended_values.mean(1)

class WhisperPhi(pl.LightningModule):
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
        self.save_hyperparameters()
    
    def configure_model(self):
        self.audio_encoder = AudioEncoder()
        self.projector = nn.Linear(1280, 2048)
        self.llm = LLM(**self.kwargs)

        self._freeze()

        self.space_indices = torch.tensor([17120, 41497, 9270, 19456, 4915, 19456]).to("cuda")
        class_counts = [1216, 733, 425, 378, 358, 22]

        total_samples = sum(class_counts)
        class_weights = [total_samples / count for count in class_counts]
        weights = torch.tensor(class_weights)
        self.loss_fn = torch.nn.CrossEntropyLoss(weights)
        self.accuracy = tm.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = MulticlassF1Score(num_classes=self.num_classes, average="macro")

    def _freeze(self):
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    
        for param in self.llm.parameters():
            param.requires_grad = False

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
        return x

    def forward(self, audio, text_tokens, text_attn_mask):
        if self.batch_size!=1:
            audio = self._pad_audio(audio)
        encoder_outputs = self.audio_encoder(audio)
        audio_features = encoder_outputs[0]
        audio_features = self.projector(audio_features)
        blank_token_position = (text_tokens[0]==109).nonzero(as_tuple=True)[0][0].item()
        text_features = self.llm.encode(torch.stack(text_tokens))
        combined_feautes, attn_mask = self._combine_features(audio_features, text_features, blank_token_position, text_attn_mask.to(device=audio_features.device))
        logits = self.llm(combined_feautes, attn_mask)
        return logits
    
    def predict_step(self, batch, batch_idx):
        preds, emotions_logits ,_ ,_ ,_ = self._get_preds_loss_accuracy(batch)
        output = [preds,emotions_logits]
        return output

    #def on_train_epoch_end(self):
    #    if self.lr_scheduler == "linear":
    #        self.hparams.lr = self.hparams.lr*self.lr_decay

    def training_step(self, batch):
        preds, logits, loss, acc, f1 = self._get_preds_loss_accuracy(batch)

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
        self.log_dict(
            {"train/acc": acc, "train/f1": f1}, #self.hparams.lr
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return {
            "loss": loss,
            "acc": acc,
            "f1": f1,
            "logits": logits,
            "preds": preds,
            "batch_size": self.batch_size,
        }

    def validation_step(self, batch, batch_idx):

        preds, logits, loss, acc, f1 = self._get_preds_loss_accuracy(batch)
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
        self.log_dict(
            {"val/acc": acc, "val/f1": f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return {"loss": loss, "acc": acc, "f1": f1, "logits": logits, "preds": preds, "batch_size": self.batch_size}

    def test_step(self, batch):
        preds, logits, loss, acc, f1 = self._get_preds_loss_accuracy(batch)

        self.log_dict(
            {"test/loss": loss, "test/acc": acc, "test/f1": f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return {"loss": loss, "acc": acc, "f1": f1, "logits": logits, "preds": preds}

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.lr
                                         )
        if self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), #FusedAdam
                                          lr=self.lr, 
                                          betas=[0.9, 0.98],
                                          eps=1e-5,
                                          weight_decay=0.01
                                          )
        if self.lr_scheduler == "linear":
            scheduler = lr_scheduler.LinearLR(optimizer, 
                                              start_factor=self.start_factor, 
                                              end_factor=self.end_factor, 
                                              total_iters=self.max_epochs
                                              )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "name":"lr_scheduler"}]
        return [optimizer]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def _get_preds_loss_accuracy(self, batch):
        audio, text_tokens, y = batch
        
        if self.batch_size!=1:
            text_tokens, text_attn_mask = self._pad_transcript(text_tokens)
        else:
            text_tokens = text_tokens[0]
            text_attn_mask = None
            
        logits = self(audio, text_tokens, text_attn_mask)

        next_token_logits = logits[:, -1, :]
        emotions_logits = next_token_logits[:, self.space_indices.to(next_token_logits.device)]
        preds = torch.argmax(emotions_logits, dim=1)
        loss = self.loss_fn(emotions_logits, y)
        acc = self.accuracy(emotions_logits, y)
        f1 = self.f1_score(emotions_logits, y)
        return preds, emotions_logits, loss, acc, f1
