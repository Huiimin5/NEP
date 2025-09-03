# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy

from torchvision.utils import save_image
from torchvision.io import read_image, ImageReadMode
import numpy as np
from torchvision.transforms.functional import resize,crop
from torchvision.transforms import InterpolationMode
from torchvision import transforms


import os
import json
from PIL import Image, ImageDraw, ImageFont

### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)
    print_last_step_logits_diff = False
    if print_last_step_logits_diff:
        l1, l2 = logits[:2] # l1: [120, 16384]
        print((l1[-1,:] - l2[-1,:]).abs().mean()) # first token diff
    return sample(logits, **sampling_kwargs)[0]

def prefill_with_probs(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, return_uncond  = False, 
                       **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)


    sampled_idx, sampled_probs = sample(logits,**sampling_kwargs)
    return sampled_idx, sampled_probs


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, 
                     **sampling_kwargs):
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
        
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos)
    sampled_idx, sampled_probs = sample(logits, **sampling_kwargs)
    return sampled_idx, sampled_probs



def decode_one_token_with_conf(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, 
                               return_uncond = False, **sampling_kwargs):
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos)
    sampled_idx, sampled_probs = sample(logits, **sampling_kwargs)
    sampled_conf = sampled_probs[torch.arange(sampled_idx.size(0)), sampled_idx[:,0]]

    if return_uncond:
        uncond_sampled_idx, uncond_sampled_probs = sample(uncond_logits, **sampling_kwargs)
        uncond_sampled_conf = uncond_sampled_probs[torch.arange(uncond_sampled_idx.size(0)), uncond_sampled_idx[:,0]]
    return sampled_idx, sampled_probs, sampled_conf, uncond_sampled_idx, uncond_sampled_probs, uncond_sampled_conf

def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag,
                  **sampling_kwargs
            )
            input_pos += 1
            # print(input_pos)
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs


@torch.no_grad()
def generate(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1,
             attn_mask = None,
               **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    # cond_combined: [32, 120, 2048]
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    return seq[:, T:]

@torch.no_grad()
def generate_with_decoder(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1,
             attn_mask = None, vq_model = None, qzshape = None,
               **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    elif attn_mask is not None:
        model.causal_mask = attn_mask
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    # cond_combined: [32, 120, 2048]
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    samples_ar = vq_model.decode_code(seq[:, T:], qzshape) # output value is between [-1, 1]

    return samples_ar


# T2I generation during validation
@torch.no_grad()
def generate_RLlamaGen(model, cond, max_new_tokens, attn_mask=None, cfg_scale=1.0, cfg_interval=-1, 
                      vq_model = None, qzshape =None, image_size = 256,
                      **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    model.causal_mask = attn_mask
    
    seq = torch.zeros((max_batch_size, T_new), device=device).int()

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)

    generated_tokens.insert(0, next_token)
    shuffled_generated_tokens = torch.cat(generated_tokens, dim = 1)

    batch_size, seq_len = shuffled_generated_tokens.size()
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
    unshuffled_x = torch.zeros_like(shuffled_generated_tokens)
    orders = model.orders[0]

    unshuffled_x[batch_indices, orders] = shuffled_generated_tokens
    seq[:, T:] = unshuffled_x

    samples_ar = vq_model.decode_code(seq[:, T:], qzshape) # output value is between [-1, 1]

    return samples_ar



def random_prefill_with_probs_img_cond(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float,
                       cond_idx_img: torch.Tensor = None, inference_time_orders = None,ntp_version = 2, 
                       raster_scan_mask = None, 
                       **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos, cond_idx_img = cond_idx_img, inference_time_orders = inference_time_orders, 
                          ntp_version = ntp_version, raster_scan_mask = raster_scan_mask, 
                          )
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos, cond_idx_img = cond_idx_img, inference_time_orders = inference_time_orders,
                    ntp_version = ntp_version, raster_scan_mask = raster_scan_mask, 
                    )

    sampled_idx, sampled_probs = sample(logits,  **sampling_kwargs)
    return sampled_idx, sampled_probs

def random_decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, 
                     inference_time_orders = None, ntp_version = 2,
                     raster_scan_mask = None,
                     **sampling_kwargs):
    # assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos, inference_time_orders = inference_time_orders,
                          ntp_version = ntp_version, raster_scan_mask = raster_scan_mask)

        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos,raster_scan_mask=raster_scan_mask)
    sampled_idx, sampled_probs = sample(logits, **sampling_kwargs)
    return sampled_idx, sampled_probs

def random_decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,    
    inference_time_orders = None, ntp_version = 2,
    raster_scan_mask = None,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = random_decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag,
                 inference_time_orders = inference_time_orders,
                 ntp_version = ntp_version, 
                 raster_scan_mask = raster_scan_mask,
                  **sampling_kwargs
            )
            input_pos += 1            
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs

# editing generation during training visualization
@torch.no_grad()
def generate_wo_saving_by_next_editing_token_prediction(model, cond, max_new_tokens,
                        attn_mask = None, cfg_scale=1.0, cfg_interval=-1, 
                        source_img = None, target_img = None,
                        mask_resized_1d = None,
                        vq_model = None, qzshape =None, 
                        image_size = 256, patch_size = 16,

                        raster_scan_mask = None,
                        mask_in_context = False,                        
                          **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")
    
    extra_cond_token_num = 0

    T_new = T + max_new_tokens 
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    model.causal_mask = attn_mask[:, :T_new, :T_new] # [2, 120 + 256 + 256, 120 + 256 + 256]
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)
    input_pos = torch.arange(0, T, device=device) # will be overwritten when the context contains other toksn
    
    """
    step1: obtain teacher forcing sequence for conditioning
    """
    source_img_latent, _, [_, _, source_img_rgb_code_seq] = vq_model.encode(source_img)
    source_img_rgb_code_seq = source_img_rgb_code_seq.unsqueeze(0)

    target_img_latent, _, [_, _, target_img_rgb_code_seq] = vq_model.encode(target_img)
    target_img_rgb_code_seq = target_img_rgb_code_seq.unsqueeze(0)

    # initialize seq with source image tokens
    seq[:, T :] = source_img_rgb_code_seq

    """
    step2: expanding conditioning
    """

    if model.model_type == 'c2i':
        raise NotImplementedError
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_img = source_img_rgb_code_seq  #[bsz, 256]
            cond_combined_img = torch.cat([cond_img,cond_img])
        else:
            cond_img = source_img_rgb_code_seq
            cond_combined_img = cond_img
    
    """
    step3: ar generation with support to image conditioning
    """    

    mask_patch_2d = mask_resized_1d.reshape(image_size//patch_size, image_size//patch_size)
    mask_patch_4d = mask_patch_2d.unsqueeze(0).unsqueeze(0)

    # editing_mask
    mask_256 = F.interpolate(mask_patch_4d.float(), size=(image_size, image_size), mode='bicubic') 
    
    image_seq_len = source_img_rgb_code_seq.size(1) #
    raster_scan_oder_1d = torch.arange(image_seq_len, device=device).unsqueeze(0).repeat(max_batch_size, 1)
    
    if max_batch_size == 1:
        order_for_editing_1d = raster_scan_oder_1d[mask_resized_1d == 1]
        order_for_editing=order_for_editing_1d.unsqueeze(0)
    else:
        raise NotImplementedError
    
    editing_steps = (mask_resized_1d == 1).sum(dim = -1).int().item()

    """
    step4: expand kv_cache for next editing token predition
    """

    if mask_in_context:
        extra_cond_token_num = max_new_tokens # image token num; as the mask is same-sized as the source image
    model.expand_caches_for_editing(T, expanded_seq_length=extra_cond_token_num + editing_steps , dtype=model.tok_embeddings.weight.dtype)
    
    input_pos = torch.arange(0, T+max_new_tokens + extra_cond_token_num, device=device)
    assert max_batch_size == 1
    # for prefilling
    next_token, next_token_prob = random_prefill_with_probs_img_cond(model, cond_combined, input_pos, cfg_scale, cond_idx_img = cond_combined_img, 
                                                              inference_time_orders = order_for_editing, 
                                                              raster_scan_mask = raster_scan_mask,
                                                              **sampling_kwargs)
    batch_indices = torch.arange(max_batch_size, device=device)
    seq[batch_indices[:, None], T+order_for_editing[:, 0]] = next_token.int()

    # for next editing token prediction
    input_pos = torch.tensor([T+max_new_tokens + extra_cond_token_num], device=device, dtype=torch.int)
    generated_tokens, _ = random_decode_n_tokens(model, next_token, input_pos, editing_steps-1, cfg_scale, cfg_interval, inference_time_orders=order_for_editing,
                                              raster_scan_mask = raster_scan_mask,
                                              **sampling_kwargs)
    if max_batch_size == 1:
        seq[:, T + order_for_editing[0, 1:]] = torch.cat(generated_tokens, dim=1).int()
    else:
        raise NotImplementedError

    samples_source_reconstruct = vq_model.decode_code(source_img_rgb_code_seq, qzshape)
    samples_target_pred = vq_model.decode_code(seq[:, T:], qzshape) # output value is between [-1, 1]
    samples_target_gt = vq_model.decode_code(target_img_rgb_code_seq, qzshape) 
    samples = torch.cat([samples_source_reconstruct,  samples_target_pred, samples_target_gt, (mask_256.repeat([1,3,1,1]) - 0.5)*2], dim = -1)

    return samples





@torch.no_grad()
def generate_magicbrush_by_next_editing_token_prediction(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, 
                           vq_model = None, qzshape =None, image_size = 256, patch_size = 16,
                          source_img_path = "none", edit_region_mask_path = None, enforce_validation_fake_masking = False,
                          save_output_img_path = 'none',
                          attn_mask = None,
                          with_mask_embedding = False,
                          mask_in_context = False,
                          **sampling_kwargs):

    GRAY = 0

    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    else:
        model.causal_mask[:] = attn_mask
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)
    input_pos = torch.arange(0, T, device=device)
    
    """
    step1: obtain the source image; init the output sequence
    """
    if isinstance(source_img_path, str): # for magicbrush images
        source_img = read_image(source_img_path,mode=ImageReadMode.RGB).unsqueeze(0).cuda()
        

        source_img = resize(source_img, [image_size, image_size])
        source_img = (source_img/255 - 0.5)*2

    else: # for emuedit test, source_img_path is  list of Image.Image objects
        source_img_tensors_li = []
        for source_img in source_img_path:

            assert isinstance(source_img, Image.Image)
            source_img = transforms.ToTensor()(source_img).unsqueeze(0).cuda()
            source_img = (source_img - 0.5)*2
            source_img_tensors_li.append(source_img)
        source_img_tensors = torch.cat(source_img_tensors_li, dim = 0)        
        source_img = source_img_tensors
    source_img_latent, _, [_, _, source_img_rgb_code_seq] = vq_model.encode(source_img)
    source_img_rgb_code_seq = source_img_rgb_code_seq.reshape(source_img.shape[0], -1)
    seq[:, T:] = source_img_rgb_code_seq


    """
    step2: expanding conditioning
    """

    if model.model_type == 'c2i':
        raise NotImplementedError
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_img = source_img_rgb_code_seq  #[bsz, 256]
            cond_combined_img = torch.cat([cond_img,cond_img])

        else:
            cond_img = source_img_rgb_code_seq
            cond_combined_img = cond_img
    
    """
    step3: ar generation with support to image conditioning

    edit_region_mask_path
    """
    if not enforce_validation_fake_masking: # by default: mask from mask file
        
        edit_region_mask_white = read_image(edit_region_mask_path,mode=ImageReadMode.RGB).unsqueeze(0).cuda()
        edit_region_mask_gray = read_image(edit_region_mask_path.replace('_white',''),mode=ImageReadMode.RGB).unsqueeze(0).cuda()

    else: # mask from fake masking; equivalant to no masking; equivalant to NTP
        edit_region_mask = torch.ones_like(source_img) * 255

    uniform_size = 512
    if not enforce_validation_fake_masking: # when mask is from mask file (white & gray)
        mask_1c = (edit_region_mask_white == 255).all(1,keepdim=True) * (edit_region_mask_gray == GRAY).all(1,keepdim=True) # all three channels are 255: white pixels
    else:
        mask_1c = (edit_region_mask == 255).all(1,keepdim=True) # all three channels are 255: white pixels

    mask_upsampling = F.interpolate(mask_1c.float(), size=(uniform_size, uniform_size), mode='bicubic') # 0-1
        
    # Downsampling by taking the maximum of each latent_size * latent_size block
    mask_ds = F.max_pool2d(mask_upsampling, kernel_size=(uniform_size// image_size * patch_size,
                                            uniform_size//image_size * patch_size))

    # sequential masking
    mask_resized_1d = (mask_ds.squeeze(1).flatten(1) > 0).int()
    image_seq_len = source_img_rgb_code_seq.size(1) #
    raster_scan_oder_1d = torch.arange(image_seq_len, device=device).unsqueeze(0).repeat(max_batch_size, 1)
    editing_steps = (mask_resized_1d == 1).sum(dim = -1)[0].int().item() # either max_batch_size == 1 or all images in current batch have the same editing steps (full image editing)

    if max_batch_size == 1:
        order_for_editing_1d = raster_scan_oder_1d[mask_resized_1d == 1]
        order_for_editing=order_for_editing_1d.unsqueeze(0)
        
    else:
        assert enforce_validation_fake_masking
        # when all samples conduct raster scan editing, the model can do batch-wise editing
        order_for_editing = raster_scan_oder_1d
                

    """
    step4: expand kv_cache for next editing token predition
    """
    extra_cond_num = 0 
    if mask_in_context:
        extra_cond_num = max_new_tokens # for accepting mask tokens in the context
    model.expand_caches_for_editing(T, expanded_seq_length=editing_steps + extra_cond_num, dtype=model.tok_embeddings.weight.dtype)        
    
    input_pos = torch.arange(0, T+max_new_tokens + extra_cond_num, device=device)
    assert max_batch_size == 1 or enforce_validation_fake_masking
    # for prefilling
    if with_mask_embedding or mask_in_context: # add mask embedding to input or mask in context
        raster_scan = torch.arange(max_new_tokens, device=device).unsqueeze(0).repeat(max_batch_size, 1)
        if cfg_scale > 1.0:
            raster_scan_mask = mask_resized_1d.repeat(2,1)
    else: # editing
        raster_scan_mask = None
    
    
    next_token, next_token_prob = random_prefill_with_probs_img_cond(model, cond_combined, input_pos, cfg_scale, cond_idx_img = cond_combined_img, 
                                                            inference_time_orders = order_for_editing, 
                                                            raster_scan_mask = raster_scan_mask,

                                                            **sampling_kwargs)
    batch_indices = torch.arange(max_batch_size, device=device)
    seq[batch_indices[:, None], T+order_for_editing[:, 0]] = next_token.int()

    # for next editing token prediction
    input_pos = torch.tensor([T+max_new_tokens+extra_cond_num], device=device, dtype=torch.int)
    generated_tokens, _ = random_decode_n_tokens(model, next_token, input_pos, editing_steps-1, cfg_scale, cfg_interval, inference_time_orders=order_for_editing,
                                            raster_scan_mask = raster_scan_mask,

                                            **sampling_kwargs)
    if max_batch_size == 1:
        seq[:, T + order_for_editing[0, 1:]] = torch.cat(generated_tokens, dim=1).int()
    else:
        assert enforce_validation_fake_masking
        # full image editing
        seq[:, T + 1:] = torch.cat(generated_tokens, dim=1).int()

    samples_target_pred = vq_model.decode_code(seq[:, T:], qzshape) # output value is between [-1, 1]
    mask_image_size = F.interpolate(mask_ds.float(), size=(image_size, image_size), mode='bicubic') # 0-1
    samples_target_pred[(mask_image_size == 0).repeat(1,3,1,1)] = source_img[(mask_image_size == 0).repeat(1,3,1,1)]
    if max_batch_size == 1 and type(save_output_img_path) != list:
        save_image(samples_target_pred, save_output_img_path, normalize=True, value_range=(-1, 1))
    else:
        assert enforce_validation_fake_masking
        for img_i in range(max_batch_size):
            save_image(samples_target_pred[img_i:img_i+1], save_output_img_path[img_i], normalize=True, value_range=(-1, 1))
    model.clear_caches()
    return
    
