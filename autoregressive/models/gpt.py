# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from utils_.drop_path import DropPath
import math
import numpy as np
import random


from einops import rearrange
import time
import os
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize,crop
from torchvision.transforms import InterpolationMode
def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    eval_order_version_id: int = 1

    set_max_len: bool = False
    # if put source image mask in the context
    mask_in_context: bool = False






#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        self.max_batch_size = max_batch_size


    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        if len(input_pos.size()) == 1: # identical input pos across samples in current batch
            assert input_pos.shape[0] == k_val.shape[2]
            k_out = self.k_cache
            v_out = self.v_cache
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val
        
        # input_pos: [B，1], k_val: [B, H, 1, D]
        else: # different pos for batch samples where input_pos is with the size: [batch_size,1]
            assert 1 == k_val.shape[2]
            k_out = self.k_cache
            v_out = self.v_cache
            # scale k_val, v_val to positions specified by input_pos
            B, H, _, D = k_val.size()            
            # input_pos_reshaped = input_pos.unsqueeze(1).unsqueeze(-1).expand([B, H, 1, D]) # input_pos: [B,1] -> [B, H, 1, D]
            input_pos = input_pos.squeeze(-1) # [B]
            # print((k_out[torch.arange(8),:,input_pos, None]==k_val).all())
            # print((v_out[torch.arange(8),:,input_pos, None]==v_val).all())
            k_out[torch.arange(self.max_batch_size),:,input_pos, None] = k_val
            v_out[torch.arange(self.max_batch_size),:,input_pos, None] = v_val
            # k_out[torch.arange(8),None, input_pos.squeeze(-1)] = k_val.squeeze(2)
            # v_out[torch.arange(8),None, input_pos.squeeze(-1)] = v_val.squeeze(2)
            

            # k_out[:, :, input_pos] = k_val
            # v_out[:, :, input_pos] = v_val
        return k_out, v_out
    
    def filter(self, kept_pos, T, filtered_len = 1):
        # input_pos: [S], k_val: [B, H, S, D]
        if len(kept_pos.size()) == 1: # identical input pos across samples in current batch
            # k_out = self.k_cache
            # v_out = self.v_cache
            # k_out[:, :, input_pos] = k_val
            # v_out[:, :, input_pos] = v_val
            raise NotImplementedError
        
        # input_pos: [B，1]
        else: # different pos for batch samples where input_pos is with the size: [batch_size,1]
            # k_out = self.k_cache
            # v_out = self.v_cache

            k_out_text, k_out = self.k_cache[:, :, :T], self.k_cache[:, :, T:]
            v_out_text, v_out = self.v_cache[:, :, :T], self.v_cache[:, :, T:]
            B,H,L,D = k_out.size() 

            # kept_pos: [B, seq_len] -> [B, H, seq_len, D]
            prefix_len =  kept_pos.size(-1) + filtered_len
            kept_pos = kept_pos.unsqueeze(1).unsqueeze(-1).repeat(1, H, 1, D)
            k_out_prefix, k_out_postfix = k_out.split([prefix_len, L - prefix_len], dim = -2)
            k_out_prefix_kept = k_out_prefix.gather(-2, kept_pos)
            k_out = torch.cat([k_out_prefix_kept, torch.zeros([B, H, filtered_len, D], device=k_out.device, dtype=k_out.dtype), k_out_postfix], dim = -2)

            v_out_prefix, v_out_postfix = v_out.split([prefix_len, L - prefix_len], dim = -2)
            v_out_prefix_kept = v_out_prefix.gather(-2, kept_pos)
            v_out = torch.cat([v_out_prefix_kept, torch.zeros([B, H, filtered_len, D], device=v_out.device, dtype=v_out.dtype), v_out_postfix], dim = -2)
        self.k_cache = torch.cat([k_out_text, k_out], dim = 2)
        self.v_cache = torch.cat([v_out_text, v_out], dim = 2) 
        # return k_out, v_out
    def expand(self, expanded_seq_length):
        max_batch_size, n_head, old_seq_length, head_dim = self.k_cache.size()
        k_cache_expanded = torch.zeros([max_batch_size, n_head, expanded_seq_length, head_dim], dtype = self.k_cache.dtype, device=self.k_cache.device)
        v_cache_expanded = torch.zeros([max_batch_size, n_head, expanded_seq_length, head_dim], dtype = self.v_cache.dtype, device=self.v_cache.device)
        self.k_cache = torch.cat([self.k_cache, k_cache_expanded], dim = 2)
        self.v_cache = torch.cat([self.v_cache, v_cache_expanded], dim = 2)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
        rope_exclude_token_num = 120,
    ):
        bsz, seqlen, embed_dim = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
    

        xq = corrected_apply_batchified_rotary_emb(xq, freqs_cis, rope_exclude_token_num = rope_exclude_token_num)
        xk = corrected_apply_batchified_rotary_emb(xk, freqs_cis, rope_exclude_token_num = rope_exclude_token_num)
            
            
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention( # output: [bsz, 20, 375, 64]
        xq, keys, values, # training: xq, xk, xv:[bsz, 20, 375, 64], 
        attn_mask=mask, 
        is_causal=True if mask is None else False, # is_causal=False is for KV cache
        dropout_p=self.attn_dropout_p if self.training else 0)     
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None, 
        rope_exclude_token_num = 120):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask, rope_exclude_token_num))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.class_dropout_prob = config.class_dropout_prob
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim) # (16384, 1280)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1


        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype,grid_times =1,  wo_kv_cache = False):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        
        if not wo_kv_cache: # kv cache not allowe
            for b in self.layers:
                b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size # whm: why this insertion????
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, grid_times=grid_times)
    
    
    
    def expand_caches(self, max_batch_size, expanded_seq_length, dtype,grid_times =1,  wo_kv_cache = False):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        # head_dim = self.config.dim // self.config.n_head
        # max_seq_length = find_multiple(max_seq_length, 8)
        # self.max_seq_length = max_seq_length
        # self.max_batch_size = max_batch_size
        
        if not wo_kv_cache: # kv cache not allowe
            for b in self.layers:
                # b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)
                b.attention.kv_cache.expand(expanded_seq_length)
        
        new_seq_length = b.attention.kv_cache.k_cache.size(2)
        old_seq_length = new_seq_length - expanded_seq_length
        causal_mask_new = torch.tril(torch.ones(new_seq_length, new_seq_length, dtype=torch.bool, device =b.attention.kv_cache.k_cache.device ))
        causal_mask_new = causal_mask_new.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        # self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        # horizontally expand
        self.causal_mask = torch.cat([self.causal_mask, causal_mask_new[:,:old_seq_length,old_seq_length:new_seq_length]], dim = 2)
        # vertically expand
        self.causal_mask = torch.cat([self.causal_mask, causal_mask_new[:,old_seq_length:new_seq_length, :]], dim = 1)
        self.freqs_cis = torch.cat([self.freqs_cis, self.freqs_cis[-expanded_seq_length:]])
        # grid_size = int(self.config.block_size ** 0.5)
        # assert grid_size * grid_size == self.block_size # whm: why this insertion????
        # self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, grid_times=grid_times)

    def clear_caches(self,):
        for b in self.layers:
            b.attention.kv_cache = None

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        cond_idx_img: Optional[torch.Tensor] = None, # + suppor to image conditioning
        diff_text_ids: Optional[torch.Tensor] = None,
    ):
        
        
        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)

            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            # if len(input_pos.size()) == 2:
            #     import pdb
            #     pdb.set_trace()

            if cond_idx_img is None: # t2i: by default, without conditioning on any images
                # import pdb
                # pdb.set_trace()
                if cond_idx is not None: # prefill in inference
                    token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num] # [4, 120, 2048] -> [4, 120, 1280]
                else: # decode_n_tokens(kv cache) in inference

                    token_embeddings = self.tok_embeddings(idx)
            else: # prifill in inference; image tokens as conditioning
                if cond_idx is not None: # prefill (t+i) in inference
                    cond_txt_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                    cond_img_embeddings = self.tok_embeddings(cond_idx_img)
                    token_embeddings = torch.cat([cond_txt_embeddings, cond_img_embeddings], dim = 1)
                else:
                    token_embeddings = self.tok_embeddings(idx)


                
            bs = token_embeddings.shape[0]
            if len(input_pos.size())==1:
                mask = self.causal_mask[:bs, None, input_pos] 
                #without prefilling image tokens [8, 1, 120, 376] or [8, 1, 1, 376] for prifilling and image token inference
                # with prefilling image tokens: [2, 1, 376, 632]
            else:
                mask = self.causal_mask[:bs][torch.arange(bs), None, input_pos.squeeze(), None] # [8, 1, 1, 376]

            h = self.tok_dropout(token_embeddings)# [8, 1280]
            self.freqs_cis = self.freqs_cis
        if self.training:
            if cond_idx_img is None: # by default
                freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
                rope_exclude_token_num = self.cls_token_num
            else:
                raise NotImplementedError
        else:
            freqs_cis = self.freqs_cis[input_pos] # [376, 32, 2] -> [120, 32, 2]; [4,1,32,2]

            if cond_idx is not None and idx is None: # prefilling
                rope_exclude_token_num = self.cls_token_num 
            else:
                rope_exclude_token_num = 0

        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask, diff_text_ids, rope_exclude_token_num = rope_exclude_token_num)

        # output layers
        h = self.norm(h)


        logits = self.output(h).float()
        
        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            # torch.set_printoptions(profile="full")
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # try:
        #     assert (logits[0] == logits[1]).all()
        #     assert (logits[2] == logits[3]).all()
        # except:
        #     import pdb
        #     pdb.set_trace()

        return logits, loss


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class RTransformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim) # (16384, 1280)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):            
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        # whm: for randomized order training
        self.image_seq_len = grid_size * grid_size
        self.grid_size = grid_size
        embed_dim = config.dim
        

        self.target_aware_pos_embed_postfix = nn.init.trunc_normal_(
        nn.Parameter(torch.zeros(1, self.image_seq_len, embed_dim)), 0., 0.02)

        self.eval_order_version_id = config.eval_order_version_id

        self.set_max_len = config.set_max_len

        # [text, source, mask tokens] -> editing tokens
        self.mask_in_context  = config.mask_in_context
        if self.mask_in_context:
            self.unmask_embedding = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, 1, embed_dim)), 0., 0.02)
            self.mask_embedding = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, 1, embed_dim)), 0., 0.02)

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def clear_caches(self,):
        for b in self.layers:
            b.attention.kv_cache = None

    def setup_caches(self, max_batch_size, max_seq_length, dtype,grid_times =1,  wo_kv_cache = False):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        
        if not wo_kv_cache: # kv cache not allowe
            for b in self.layers:
                b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size # whm: why this insertion????
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, grid_times=grid_times)

    def expand_caches_for_editing(self, T, expanded_seq_length, dtype,grid_times =1,  wo_kv_cache = False):
        
        if not wo_kv_cache: # kv cache not allowe
            for b in self.layers:
                # b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)
                b.attention.kv_cache.expand(expanded_seq_length)
        new_seq_length = b.attention.kv_cache.k_cache.size(2)
        old_seq_length = new_seq_length - expanded_seq_length
        causal_mask_new = torch.tril(torch.ones(new_seq_length, new_seq_length, dtype=torch.bool, device=b.attention.kv_cache.k_cache.device))
        causal_mask_new = causal_mask_new.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        # new q to old kv, the same as the original causal mask
        causal_mask_new[:, :old_seq_length, :old_seq_length] =self.causal_mask
        # new q to old kv = last old q repeat
        causal_mask_new[:, old_seq_length:new_seq_length, :old_seq_length] =self.causal_mask[:,old_seq_length-1: old_seq_length].repeat(1, expanded_seq_length, 1)
        self.causal_mask = causal_mask_new

        # for new queries to old kv, the same as the original causal mask
    def get_raster_orders(self, x):
        batch_size = x.shape[0]
        shuffled_orders = torch.stack([torch.arange(self.image_seq_len, device=x.device) for _ in range(batch_size)])
        return shuffled_orders

    def get_reverse_raster_orders(self, x):
        batch_size = x.shape[0]
        shuffled_orders = torch.stack([torch.arange(self.image_seq_len - 1, -1, -1, device=x.device) for _ in range(batch_size)])
        return shuffled_orders

    def shuffle(self, x, orders):
        batch_size, seq_len = x.shape[:2]

        order_seq_len = orders.size(-1)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, order_seq_len)
        shuffled_x = x[batch_indices, orders]
        return shuffled_x

    def unshuffle(self, shuffled_x, orders):
        # Unshuffle the tensor based on the original orders
        batch_size, seq_len = shuffled_x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x

    def sample_orders(self, x):
        batch_size = x.shape[0]
        shuffled_orders = []

        for _ in range(batch_size):
            shuffled_orders.append(torch.randperm(self.image_seq_len, device=x.device))
                
        shuffled_orders = torch.stack(shuffled_orders)
        return shuffled_orders.to(x.device)

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        cond_idx_img: Optional[torch.Tensor] = None, # + suppor to image conditioning
        inference_time_orders = None, 
        ntp_version = 1,
        return_orders = False,

        generation_orders_for_editing = None,
        raster_scan_mask = None, # editing regions first, unedited regions last
        editing_mask = None, # [batch_size, image_seq_len]

        with_cfg = True,
    ):
        
        if self.training:
            if cond_idx_img is None: # for t2i without source image tokens
                orders = self.sample_orders(idx)

            else: # for editing with source image tokens
                orders_full = generation_orders_for_editing# editing regions first, unedited regions last
                batch_max_editing_region = torch.nonzero(editing_mask.sum(0)) # [image_sequence_length, 1]
                if batch_max_editing_region.size(0) == 0:
                    print('a batch of non-editing image')
                    return None, torch.zeros(1, device = idx.device)
                mask_generation_step = batch_max_editing_region[-1,0].item() if not self.set_max_len else self.image_seq_len - 1
                orders = orders_full[:, :mask_generation_step + 1]
                editing_mask_full = editing_mask
                editing_mask = editing_mask_full[:, :mask_generation_step + 1]


        else:
            if inference_time_orders is None: # raster scan order generation
                if idx is None: # whm: for prefilling
                    orders = self.get_raster_orders(cond_idx)
                elif cond_idx is None:
                    orders = self.get_raster_orders(idx)                
            else: # generation by user-specified order
                orders = inference_time_orders
            
        
        self.orders  = orders

        if idx is not None and cond_idx is not None: # training or naive inference
            if cond_idx_img is not None: # for editing
                token_embeddings_cond_image = self.tok_embeddings(cond_idx_img)
                cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                token_embeddings = self.tok_embeddings(idx)
                token_embeddings = torch.cat((cond_embeddings, token_embeddings_cond_image, token_embeddings), dim=1)
                h = self.tok_dropout(token_embeddings)

            else: # for t2i
                cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                token_embeddings = self.tok_embeddings(idx)
                token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
                h = self.tok_dropout(token_embeddings)

            self.freqs_cis = self.freqs_cis.to(h.device)
            bs = idx.size(0) if idx is not None else cond_idx.size(0)


        else:
            # for t2i
            if cond_idx_img is None: # by default, without conditioning on any images
                if cond_idx is not None: # prefill in inference
                    token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num] # [4, 120, 2048] -> [4, 120, 1280]
                else: # decode_n_tokens(kv cache) in inference
                    token_embeddings = self.tok_embeddings(idx)

            # for editing
            else: #  prifill in inference; image tokens as conditioning;                
                if cond_idx is not None: # prefill (t+i) in inference
                    cond_txt_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                    cond_img_embeddings = self.tok_embeddings(cond_idx_img)
                    token_embeddings = torch.cat([cond_txt_embeddings, cond_img_embeddings], dim = 1)
                else:
                    token_embeddings = self.tok_embeddings(idx)
            bs = token_embeddings.size(0)
            
            if len(input_pos.size())==1:
                mask = self.causal_mask[:bs, None, input_pos] 
            else:
                mask = self.causal_mask[:bs][torch.arange(bs), None, input_pos.squeeze(), None] # [8, 1, 1, 376]
            h = self.tok_dropout(token_embeddings)# [8, 1280]
            self.freqs_cis = self.freqs_cis
        
        # shuffle embeddings, targets for training
        if self.training:
            targets = self.shuffle(targets, orders)
            if cond_idx_img is None: # by default for t2i
                conditioning_token_num = self.cls_token_num
            else: # for editing
                conditioning_token_num = self.cls_token_num + self.image_seq_len
            
            # cond is text for t2i and txt + source image for editing
            input_h_cond = h[:, :conditioning_token_num] 
            
            # output image tokens
            input_h_img = self.shuffle(h[:, conditioning_token_num:], orders)[:,:-1] # the last shuffled token is not used for training
            h = torch.cat([input_h_cond, input_h_img], dim=1)

            # prepare positional embeddings.
            pos_embed = self.freqs_cis.unsqueeze(0).repeat(bs, 1, 1, 1) # torch.Size([2, 376, 32, 2])

            if cond_idx_img is None: # t2i
                pos_embed_prefix = pos_embed[:, :self.cls_token_num] # text pe for t2i; 
                pos_embed_postfix_raster_scan = pos_embed[:, self.cls_token_num:]
            else: # edting
                pos_embed_prefix = pos_embed[:, :conditioning_token_num] #  text + source image pe for editing
                pos_embed_postfix_raster_scan = pos_embed[:, self.cls_token_num: conditioning_token_num]

            pos_embed_postfix = self.shuffle(pos_embed_postfix_raster_scan, orders) 
            pos_embed = torch.cat([pos_embed_prefix, pos_embed_postfix], dim=1)

            target_aware_pos_embed_raster_scan = self.target_aware_pos_embed_postfix[:,:self.image_seq_len].repeat(bs, 1, 1)
            target_aware_pos_embed_postfix = self.shuffle(target_aware_pos_embed_raster_scan, orders)
            
            if cond_idx_img is None: # for t2i only apply target aware pe to target image
                target_aware_pos_embed = torch.cat([
                    torch.zeros_like(h[:, :self.cls_token_num-1]),
                    target_aware_pos_embed_postfix,
                ], dim=1) # [bsz, 375 (120+255), 1280] for t2i; 
            else:  # for editing, apply target-aware PE to source image & editing targets
                target_aware_pos_embed = torch.cat([
                    torch.zeros_like(h[:, :self.cls_token_num-1]),
                    target_aware_pos_embed_raster_scan, # this way editing can be considered as prmopting t2i
                    target_aware_pos_embed_postfix,
                ], dim=1)
        else:
            
            if ntp_version == 1:
                if cond_idx_img is None: # without conditioning on img
                # prepare positional embeddings.
                # shuffle pos embed:  permute does not impact text tokens
                    pos_embed = self.freqs_cis.unsqueeze(0).repeat(bs, 1, 1, 1) # self.freqs_cis: [376, 32, 2]
                    pos_embed_prefix = pos_embed[:, :self.cls_token_num]
                    pos_embed_raster_scan = pos_embed[:, self.cls_token_num:]
                    pos_embed_postfix = self.shuffle(pos_embed_raster_scan, orders) 
                    pos_embed = torch.cat([pos_embed_prefix, pos_embed_postfix], dim=1)

                    target_aware_pos_embed_raster_scan = self.target_aware_pos_embed_postfix.repeat(bs, 1, 1)
                    target_aware_pos_embed_postfix = self.shuffle(target_aware_pos_embed_raster_scan, orders)
                        
                    target_aware_pos_embed = torch.cat([
                            torch.zeros(h.size(0), self.cls_token_num-1, h.size(2), device=h.device, dtype=h.dtype),
                            target_aware_pos_embed_postfix,
                            ], dim=1) # [bsz, 375 (120+255), 1280]


                    conditioning_token_num = self.cls_token_num

            elif ntp_version == 2: # editing: relying on source image
                # prepare positional embeddings.
                # shuffle pos embed: te permute does not impact text tokens
                pos_embed = self.freqs_cis.unsqueeze(0).repeat(bs, 1, 1, 1) # self.freqs_cis: [376, 32, 2]
                pos_embed_prefix = pos_embed[:, :self.cls_token_num]
                pos_embed_postfix = pos_embed[:, self.cls_token_num:]

                if orders.size(0) == 1:
                    pos_embed_editing = self.shuffle(pos_embed_postfix, orders[0]) # torch.Size([2, 256, 32, 2])
                else:
                    orders_ = orders.unsqueeze(2).unsqueeze(2).repeat(1,1 , pos_embed_postfix.size(2), pos_embed_postfix.size(3))
                    if with_cfg:
                        orders_ = orders_.repeat(2, 1, 1, 1)
                    pos_embed_editing = pos_embed_postfix.gather(1, orders_)
                
                pos_embed = torch.cat([pos_embed, pos_embed_editing], dim=1)

                target_aware_pos_embed_raster_scan = self.target_aware_pos_embed_postfix[:, :self.image_seq_len].repeat(bs, 1, 1)
                target_aware_pos_embed_editing = self.shuffle(target_aware_pos_embed_raster_scan, orders[0])

                conditioning_token_num = self.cls_token_num + self.image_seq_len
                target_aware_pos_embed = torch.cat([
                    torch.zeros(h.size(0), self.cls_token_num-1, h.size(2), device=h.device, dtype=h.dtype),
                    target_aware_pos_embed_raster_scan,# prompting t2i for editing
                    target_aware_pos_embed_editing, 
                ], dim=1) # [bsz, 377 (120+255+2 eidting token), 1280]
                
        if self.mask_in_context: # insert mask into context, update h, pos_embed and target_aware_pos_embed
            if self.training:
                raster_scan_to_pos = generation_orders_for_editing.argsort(dim = -1) # [bsz, 256]
                raster_scan_mask = editing_mask_full.gather(1, raster_scan_to_pos) # [bsz, 256]
            else:
                raster_scan_mask = raster_scan_mask

            is_editing_training = self.training and cond_idx_img is not None
            is_editing_prefilling = not self.training and cond_idx_img is not None and input_pos.size(0) > 1
            if is_editing_training or is_editing_prefilling:
                # 1. insert mask into h
                h_txt, h_source, h_editing = h.split([self.cls_token_num, self.image_seq_len, h.size(1) - self.cls_token_num - self.image_seq_len], dim = 1)
                h_mask = self.unmask_embedding.repeat(h_source.size(0), h_source.size(1), 1) # [bsz, seq_len, embed_dim]
                h_mask = h_mask.masked_scatter_(raster_scan_mask.unsqueeze(-1).repeat(1,1,self.config.dim).bool(), self.mask_embedding.expand_as(h_mask))
                h = torch.cat([h_txt, h_source, h_mask, h_editing], dim = 1)
                # count mask embeddings as part of the context/condition
                conditioning_token_num += self.image_seq_len
            
            # conduct the following no matter training or inference
            # 2. insert pos_embed corresponding to mask 
            pos_embed_txt, pos_embed_source, pos_embed_editing = pos_embed.split([self.cls_token_num, self.image_seq_len, pos_embed.size(1) - self.cls_token_num - self.image_seq_len], dim = 1)
            pos_embed_mask = pos_embed_source
            pos_embed = torch.cat([pos_embed_txt, pos_embed_source, pos_embed_mask, pos_embed_editing], dim = 1)
            # 3. insert target-aware pos_embed corresponding to mask
            target_aware_pos_embed_txt, target_aware_pos_embed_source, target_aware_pos_embed_editing = target_aware_pos_embed.split([self.cls_token_num - 1, self.image_seq_len, target_aware_pos_embed.size(1) - self.cls_token_num - (self.image_seq_len - 1)], dim = 1)        
            target_aware_pos_embed_mask = target_aware_pos_embed_source
            target_aware_pos_embed = torch.cat([target_aware_pos_embed_txt, target_aware_pos_embed_source, target_aware_pos_embed_mask, target_aware_pos_embed_editing], dim = 1)

        if not self.training:
            target_aware_pos_embed = target_aware_pos_embed.permute(1,0,2)[input_pos].permute(1,0,2)
            
        
        h = h + target_aware_pos_embed

        if self.training:
            if cond_idx_img is None: # by default t2i
                pos_embed = pos_embed[:, :h.shape[1]] # [4, 376, 32, 2]
            else: # editing
                pos_embed = pos_embed[:, :h.shape[1]]
                mask = mask[:, :, :h.shape[1], :h.shape[1]]  # truncate mask 
                targets = targets[:, :orders.shape[1] ] # truncate targets
            rope_exclude_token_num = self.cls_token_num

        else:
            pos_embed_batch_wise = pos_embed.permute(1,0,2,3)[input_pos].permute(1,0,2,3)
            pos_embed = pos_embed_batch_wise

            if cond_idx is not None and idx is None: # prefilling
                rope_exclude_token_num = self.cls_token_num 
            else:
                rope_exclude_token_num = 0
    
        for layer_id, layer in enumerate(self.layers):
            h = layer(h, pos_embed, input_pos, mask, rope_exclude_token_num = rope_exclude_token_num)
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        
        if self.training:
            if cond_idx_img is None: # for t2i training
                logits = logits[:, self.cls_token_num - 1:].contiguous()
            else: # for editing
                logits = logits[:, conditioning_token_num -1 :].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none') # [bsz * seq_len]            
            
            if cond_idx_img is None:                
                valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            else: # for editing
                valid_all = editing_mask.reshape(-1)*valid[:,None].repeat(1, targets.shape[1]).view(-1)

            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)

            
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if not return_orders: # by default
            return logits, loss 
        else:
            return logits, loss, orders


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)

#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, grid_times = 1):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1), 
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size*grid_times, grid_size*grid_times, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size*grid_times, grid_size*grid_times, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    if grid_times > 1:
        # cat_multi_image = []
        # for grid_id in range(grid_times):
        #     cat_multi_image.append(cache_grid[grid_id*grid_size: (grid_id+1)*grid_size, grid_id*grid_size: (grid_id+1)*grid_size].flatten(0,1))
        cache = torch.cat([cache]*grid_times, dim = 0)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 

def corrected_precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, grid_times = 1):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1), 
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size*grid_times, grid_size*grid_times, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size*grid_times, grid_size*grid_times, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    if grid_times > 1:
        # cat_multi_image = []
        # for grid_id in range(grid_times):
        #     cat_multi_image.append(cache_grid[grid_id*grid_size: (grid_id+1)*grid_size, grid_id*grid_size: (grid_id+1)*grid_size].flatten(0,1))
        cache = torch.cat([cache]*grid_times, dim = 0)
    # cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cache 

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    if len(freqs_cis.size()) == 3:
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    else:
        assert len(freqs_cis.size()) == 4
        import pdb; pdb.set_trace()
        freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2) # (bsz, seq_len, 1, head_dim//2, 2)    
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def apply_batchified_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (bs, seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    assert len(freqs_cis.size()) == 4
    # import pdb; pdb.set_trace()
    freqs_cis = freqs_cis.view(xshaped.size(0), xshaped.size(1), 1, xshaped.size(3), 2) # (bs, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3) # (bs, seq_len, n_head, head_dim)
    return x_out2.type_as(x)

def corrected_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, rope_exclude_token_num=1):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    xshaped_cls, xshaped_img = xshaped.split([rope_exclude_token_num, xshaped.size(1) - rope_exclude_token_num], dim = 1)
    freqs_cis_cls, freqs_cis_img = freqs_cis.split([rope_exclude_token_num, freqs_cis.size(0) - rope_exclude_token_num], dim = 0)
    if len(freqs_cis.size()) == 3:
        freqs_cis_img = freqs_cis_img.view(1, xshaped_img.size(1), 1, xshaped_img.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    else:
        assert len(freqs_cis.size()) == 4
        import pdb; pdb.set_trace()
        freqs_cis = freqs_cis.view(-1, xshaped_img.size(1), 1, xshaped_img.size(3), 2) # (bsz, seq_len, 1, head_dim//2, 2)    
    x_out2_img = torch.stack([
            xshaped_img[..., 0] * freqs_cis_img[..., 0] - xshaped_img[..., 1] * freqs_cis_img[..., 1],
            xshaped_img[..., 1] * freqs_cis_img[..., 0] + xshaped_img[..., 0] * freqs_cis_img[..., 1],
    ], dim=-1)
    x_out2 = torch.cat([xshaped_cls, x_out2_img], dim = 1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
def corrected_apply_batchified_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, rope_exclude_token_num=120):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (bs, seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    xshaped_cls, xshaped_img = xshaped.split([rope_exclude_token_num, xshaped.size(1) - rope_exclude_token_num], dim = 1)
    freqs_cis_cls, freqs_cis_img = freqs_cis.split([rope_exclude_token_num, freqs_cis.size(1) - rope_exclude_token_num], dim = 1)
    assert len(freqs_cis.size()) == 4
    freqs_cis_img = freqs_cis_img.view(xshaped_img.size(0), xshaped_img.size(1), 1, xshaped_img.size(3), 2) # (bs, seq_len, 1, head_dim//2, 2)
    x_out2_img = torch.stack([
            xshaped_img[..., 0] * freqs_cis_img[..., 0] - xshaped_img[..., 1] * freqs_cis_img[..., 1],
            xshaped_img[..., 1] * freqs_cis_img[..., 0] + xshaped_img[..., 0] * freqs_cis_img[..., 1],
    ], dim=-1)
    x_out2 = torch.cat([xshaped_cls, x_out2_img], dim = 1)
    x_out2 = x_out2.flatten(3) # (bs, seq_len, n_head, head_dim)
    return x_out2.type_as(x)

#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        
def R_GPT_XL(**kwargs):
    return RTransformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M


GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 'R-GPT-XL': R_GPT_XL
}
