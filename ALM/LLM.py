import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from ALM.modeling_gemma import GemmaForCausalLM

class CustomGemmaConfig():
    def __init__(self):
        with open('ALM/gemma_config.json') as f:
            config = json.load(f)
        for key in config:
            setattr(self, key, config[key])

class LLM(torch.nn.Module):
    def __init__(self, **kwargs): 
        super(LLM, self).__init__()
        self.llm_name = kwargs.get("llm_name", "gemma")

        self.config = CustomGemmaConfig()
        gemma = GemmaForCausalLM(self.config)
        #gemma.initialize_parameters()
        #gemma.load_state_dict(torch.load("checkpoints/gemma_2b.pth"))
        mask = torch.full(
            (self.config.max_position_embeddings, self.config.max_position_embeddings),
            fill_value=torch.finfo(gemma.dtype).min,
        )
        self.causal_mask = torch.triu(mask, diagonal=1)

        self.embed_weight = gemma.model.embed_tokens.weight
        self.blocks = gemma.model.layers
        self.final_layer_norm = gemma.model.norm
        self.lm_head = gemma.lm_head
            
    def _update_causal_mask(self, attention_mask, input_tensor):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = torch.full((2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]), fill_value=1)
            self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)

        min_dtype = torch.finfo(dtype).min
        causal_mask = self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) #* min_dtype
        causal_mask = abs(causal_mask)*-1

        causal_mask = causal_mask.to(dtype=dtype, device=device)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if self.config._attn_implementation == "sdpa" and attention_mask is not None:
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                causal_mask = causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True)).to(dtype)

        return causal_mask

    def encode(self, x):
        if self.llm_name=='gemma':
            x = torch.nn.functional.embedding(x, self.embed_weight, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
            return x
        else:
            return self.embed_tokes(x.to(device=self.embed_tokes.weight.device))#9.to(device=self.embed_tokes.weight.device

    def forward(self, x, attention_mask=None):
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
        x = self.final_layer_norm(x[0])
        x = self.lm_head(x)
        return x #.mean(1)
