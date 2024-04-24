import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaForCausalLM, PretrainedConfig
import json

class GemmaConfig(PretrainedConfig):
    def __init__(self):
        with open('ALM/gemma_config.json') as f:
            config = json.load(f)
        for key in config:
            setattr(self, key, config[key])

class LLM(torch.nn.Module):
    def __init__(self, **kwargs): 
        super(LLM, self).__init__()
        self.llm_name = kwargs.get("llm_name", "gemma")
        if self.llm_name=='phi':
            phi = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype="auto", trust_remote_code=True)

            self.embed_tokes = phi.model.embed_tokens
            self.blocks = phi.model.layers
            self.final_layer_norm = phi.model.final_layernorm
            self.lm_head = phi.lm_head

        elif self.llm_name=='gemma':
            #self.config = GemmaConfig()
            #gemma = GemmaForCausalLM(self.config)
            gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
            self.config = gemma.config
            mask = torch.full(
                (self.config.max_position_embeddings, self.config.max_position_embeddings),
                fill_value=torch.finfo(gemma.dtype).min,
            )
            self.causal_mask = torch.triu(mask, diagonal=1)
            self._update_causal_mask = gemma.model._update_causal_mask

            self.embed_weight = gemma.model.embed_tokens.weight
            self.blocks = gemma.model.layers
            self.final_layer_norm = gemma.model.norm
            self.lm_head = gemma.lm_head
            #self.fc1 = torch.nn.Linear(2048, 512, bias=False)
            #self.bn1 = torch.nn.BatchNorm1d(512)
            #self.fc2 = torch.nn.Linear(512, 8, bias=True)
            

    def encode(self, x):
        if self.llm_name=='gemma':
            x = torch.nn.functional.embedding(x, self.embed_weight, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
            return x
        else:
            return self.embed_tokes(x.to(device=self.embed_tokes.weight.device))#9.to(device=self.embed_tokes.weight.device

    def forward(self, x, attention_mask=None):
        if self.llm_name=='gemma':
            x = x * (self.config.hidden_size**0.5)
            past_seen_tokens = 0
            cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + x.shape[1]
               )
            position_ids = cache_position.unsqueeze(0)
            causal_mask = self._update_causal_mask(attention_mask, x)
            x=(x,)
            for block in self.blocks:
                x = block(x[0],
                        position_ids=position_ids.to(x[0].device),
                        attention_mask=causal_mask,
                        cache_position=cache_position.to(x[0].device),
                        use_cache=False,
                        )
        if self.llm_name == "phi":
            x=(x,)
            for block in self.blocks:
                x = block(x[0])
        x = self.final_layer_norm(x[0])
        x = self.lm_head(x)
        #x=self.fc1(x)
        #x=self.bn1(x.permute(0,2,1))
        #x=self.fc2(x.permute(0,2,1))
        return x #.mean(1)
