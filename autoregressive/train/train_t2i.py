# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT
#   nanoGPT: https://github.com/karpathy/nanoGPT
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from glob import glob
import time
import argparse
import os
import json

from utils_.distributed import init_distributed_mode
from utils_.logger import create_logger
from dataset.build import build_dataset
from dataset.augmentation import center_crop_arr, center_crop_arr_removing_white_border
from autoregressive.train.train_c2i import creat_optimizer
from autoregressive.models.gpt import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models




from torch.utils.tensorboard import SummaryWriter
from autoregressive.models.generate import generate_RLlamaGen,generate_with_decoder
from torchvision import transforms


from evaluations.t2i.evaluation import evaluate_model_during_validation
import pandas as pd
import math
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Subset



def main(args, val_args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    if args.distributed:
        assert args.global_batch_size % dist.get_world_size() % args.gradient_accumulation_steps == 0, f"Batch size must be divisible by world size."
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
    else:
        device, rank = 0, 0
        seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_SHOW_CPP_STACKTRACES']="1"
    os.environ['TORCH_CPP_LOG_LEVEL']="INFO"




    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-") 
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        tensorboard_dir = f"{experiment_dir}/tensorboard"
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)

        logger = create_logger(experiment_dir, rank )
        logger.info(f"Experiment directory created at {experiment_dir}")

        if not args.no_cloud_save:

            time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
            cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
            os.makedirs(cloud_checkpoint_dir, exist_ok=True)
            logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")

    
    else:
        checkpoint_dir = None
        logger = create_logger(None, -1)

    if args.distributed:
        checkpoint_dir = [checkpoint_dir]
        dist.broadcast_object_list(checkpoint_dir, src=0)
        checkpoint_dir = checkpoint_dir[0]

    # training args
    logger.info(f"{args}")

    # training env
    if args.distributed:
        logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        logger.info(f"Starting rank={rank}, seed={seed}, world_size=1.")


    # Setup model
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,

    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    # Setup data:
    if args.dataset.startswith('t2i'):     # create and load model
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim)
        vq_model.to(device)
        vq_model.eval()
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        del checkpoint       
    if not args.aug_remove_white_borders:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr_removing_white_border(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    dataset = build_dataset(args, transform=transform)

    
    world_size = dist.get_world_size() if args.distributed else 1
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size // args.gradient_accumulation_steps),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images")




    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / (args.global_batch_size / args.gradient_accumulation_steps)))
        # train_steps = int(start_epoch * int(len(dataset) / (args.global_batch_size / args.gradient_accumulation_steps)))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")

    else:
        train_steps = 0
        start_epoch = 0
    if args.init_gpt_ckpt:
        checkpoint = torch.load(args.init_gpt_ckpt, map_location="cpu")
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(f"init from {args.init_gpt_ckpt}", msg)

    if args.distributed:
        model_wo_ddp = model
        model = DDP(model.to(device), device_ids=[args.gpu])
    

    model.train()  # important! This enables embedding dropout for classifier-free guidance

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0

    running_loss_ce, running_loss_proj = 0, 0
    start_time = time.time()
    exit_flag = False
    logger.info(f"Training for {args.epochs} epochs...")


    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for x, y, attn_mask, valid in loader:

            if args.early_stop != -1 and train_steps == args.early_stop:
                exit_flag = True
                break
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            if args.dataset.startswith('t2i'):
                img = x
                with torch.no_grad():
                    _, _, [_, _, indices] = vq_model.encode(img)
                x = indices.reshape(img.shape[0], -1)
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(y.shape[0], y.shape[-2], y.shape[-1])
            assert z_indices.shape[0] == c_indices.shape[0]
            attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, attn_mask.shape[-2], attn_mask.shape[-1]) # (bs, n_head, seq_len, seq_len)

            try:
                if args.gpt_model.startswith("R-GPT"): # RLlamaGen training
                    with torch.amp.autocast('cuda', dtype=ptdtype):  
                        _, loss = model(cond_idx=c_indices, idx=z_indices, targets=z_indices, mask=attn_mask[:, :, :-1,:-1], valid=valid)
                    
                else: # default ar training
                    with torch.amp.autocast('cuda', dtype=ptdtype):  
                        _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices, mask=attn_mask[:, :, :-1,:-1], valid=valid)
            except Exception as e:
                logger.error(f"Error during forward pass: {e}")
                raise
            
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            
            if (train_steps + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Log loss values:
            running_loss += loss.item()
            running_loss_ce += loss.item()
            log_steps += 1
            train_steps += 1
            
            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                        # Check and delete the oldest checkpoint if the total number exceeds the threshold
                        checkpoint_files = sorted(glob(f"{checkpoint_dir}/*.pt"), key=os.path.getctime)
                        if len(checkpoint_files) > args.max_checkpoints:
                            oldest_checkpoint = checkpoint_files[0]
                            os.remove(oldest_checkpoint)
                            logger.info(f"Deleted oldest checkpoint {oldest_checkpoint} to maintain max checkpoints limit")
                    if not args.no_cloud_save:
                        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, cloud_checkpoint_path)
                        logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")

                        # Check and delete the oldest checkpoint if the total number exceeds the threshold
                        checkpoint_files = sorted(glob(f"{checkpoint_dir}/*.pt"), key=os.path.getctime)
                        if len(checkpoint_files) > args.max_checkpoints:
                            oldest_checkpoint = checkpoint_files[0]
                            os.remove(oldest_checkpoint)
                            logger.info(f"Deleted oldest checkpoint {oldest_checkpoint} to maintain max checkpoints limit")
                dist.barrier()

            # validate by calculating fid and clip score
            if args.validate_every != -1 and train_steps % args.validate_every == 0:
                dist.barrier()
                model.eval()
                # sample
                # Create folder to save samples:
                model_string_name = args.gpt_model.replace("/", "-")
                ckpt_string_name = f"{checkpoint_dir}-{train_steps:07d}"
                prompt_name = args.prompt_csv.split('/')[-1].split('.')[0].lower()
                folder_name = f"{ckpt_string_name}-{prompt_name}-size-{args.image_size}-size-{args.image_size}-{args.vq_model}-" \
                            f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                            f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
                # sample_folder_dir = f"{args.sample_dir}/{folder_name}"
                sample_folder_dir = folder_name
                if rank == 0:
                    os.makedirs(f"{sample_folder_dir}/images", exist_ok=True)

                df = pd.read_csv(args.prompt_csv, delimiter='\t')
                prompt_list = df['Prompt'].tolist()

                val_dataset_full = build_dataset(val_args, transform=transform)
                val_global_batch_size = args.global_batch_size 
                n = args.global_batch_size // dist.get_world_size()
                val_subset = Subset(val_dataset_full, range(args.num_fid_samples))
                val_sampler = DistributedSampler(
                    val_subset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    seed=args.global_seed
                )
                val_loader = DataLoader(
                    val_subset,
                    batch_size=int(args.global_batch_size // world_size // args.gradient_accumulation_steps),
                    shuffle=False,
                    sampler=val_sampler,

                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True
                )
                num_fid_samples = min(args.num_fid_samples, len(val_subset))

                # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
                total_samples = int(math.ceil(num_fid_samples / val_global_batch_size) * val_global_batch_size)
                if rank == 0:
                    print(f"Total number of images that will be sampled: {total_samples} in {sample_folder_dir}")
                assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
                samples_needed_this_gpu = int(total_samples // dist.get_world_size())
                assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"


                total_val = 0
                validation_samples, validation_samples_gt = [], []
                for val_data in val_loader:

                    val_img, val_t5_feat_padding, val_attn_mask, _ = val_data
                    val_img, val_t5_feat_padding, val_attn_mask = val_img.to(device), val_t5_feat_padding.to(device), val_attn_mask.to(device)
                    val_t5_feat_padding = val_t5_feat_padding[:, 0] #
                    val_attn_mask = val_attn_mask[:, 0] # bsz, 1, seq_len, seq_len
                    if args.cfg_scale != 1:
                        val_attn_mask = val_attn_mask.repeat([2,1,1])
                    qzshape = [val_img.size(0), args.codebook_embed_dim, latent_size, latent_size]

                    if args.gpt_model.startswith("R-GPT") :

                        val_index_sample = generate_RLlamaGen(
                        model.module, val_t5_feat_padding, latent_size ** 2, 
                        val_attn_mask, 
                        cfg_scale=args.cfg_scale,
                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p, sample_logits=True, 
                        vq_model = vq_model, qzshape =qzshape, 
                        image_size=args.image_size,
                        
                        )
                    else:
                        val_index_sample = generate_with_decoder(
                        model.module, val_t5_feat_padding, latent_size ** 2, 
                        emb_masks=None,
                        attn_mask=val_attn_mask, 
                        cfg_scale=args.cfg_scale,

                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p, sample_logits=True, 
                        vq_model = vq_model, qzshape =qzshape,                         
                        )
                    if rank == 0:
                        # log the first image in each batch
                        val_index_sample_tb_norm = (val_index_sample[0] + 1) / 2
                        validation_samples.append(val_index_sample_tb_norm)
                        validation_samples_gt.append((val_img[0] + 1) / 2)
                    
                    samples = torch.clamp(127.5 * val_index_sample + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                    
                    # Save samples to disk as individual .png files
                    for i, sample in enumerate(samples):
                        index = i * dist.get_world_size() + rank + total_val
                        Image.fromarray(sample).save(f"{sample_folder_dir}/images/{index:06d}.png")
                    total_val += val_global_batch_size
                    model.module.clear_caches()

                # Make sure all processes have finished saving their samples before attempting to convert to .npz
                dist.barrier()
                model.train()
                if rank == 0:
                    # log generated images into tb
                    writer.add_image(f'Inference Sample', torch.cat(validation_samples, dim=-1), train_steps)
                    writer.add_image(f'Inference Sample GT', torch.cat(validation_samples_gt, dim=-1), train_steps)

                    # Save infer result in a jsonl file
                    json_items = []
                    for idx, prompt in enumerate(prompt_list):
                        image_path = os.path.join(sample_folder_dir, "images", f"{idx:06d}.png")
                        json_items.append({"text": prompt, "image_path": image_path})
                    res_jsonl_path = os.path.join(sample_folder_dir, "result.jsonl")
                    print(f"Save jsonl to {res_jsonl_path}...")
                    with open(res_jsonl_path, "w") as f:
                        for item in json_items:
                            f.write(json.dumps(item) + "\n")

                    # Save captions to txt
                    caption_path = os.path.join(sample_folder_dir, "captions.txt")
                    print(f"Save captions to {caption_path}...")
                    with open(caption_path, "w") as f:
                        for item in prompt_list:
                            f.write(f"{item}\n")
                    # evaluate


                    clip_score, fid = evaluate_model_during_validation(ref_dir="/home/wuhuimin/data/coco/images/", ref_data = 'coco2014', ref_type = 'val2014', 
                                                     fake_dir = sample_folder_dir, eval_res=args.eval_res, 
                                                    eval_batch_size=args.eval_batch_size, eval_clip_model4eval=args.eval_clip_model4eval,
                                                    clip_device = 'cpu', fid_device = torch.device('cuda:0'), use_dataparallel = False)#, how_many=total_samples)
                    print(clip_score, fid)
                    writer.add_scalar('Evaluation/CLIP', clip_score, train_steps)
                    writer.add_scalar('Evaluation/fid', fid, train_steps)
                dist.barrier()
                # end of saving
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if args.distributed:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0

                # Reset monitoring variables:
                log_steps = 0
                start_time = time.time()

                # Log loss to TensorBoard
                # sampled images to Tensorboard
                if rank == 0: 
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                    writer.add_scalar('Loss_ce/train', avg_loss, train_steps) # loss is ce loss


                    model.eval()
                    model.module.eval_order_version_id = args.eval_order_version_id
                    
                    visualization_image_ids = [0, 4, 16, 64, 128]
                    all_visualization_images = []
                    all_visualization_images_gt = []
                    for visualization_image_id in visualization_image_ids:
                        try:
                            data = dataset[visualization_image_id]
                        except Exception as e:
                            print(e)
                            continue
                        img, t5_feat_padding, attn_mask, _ = data # t5_feat_padding.size()
                        img, t5_feat_padding, attn_mask = img.to(device), t5_feat_padding.to(device), attn_mask.to(device)

                        qzshape = [1, args.codebook_embed_dim, latent_size, latent_size]

                        if args.cfg_scale != 1:
                            attn_mask = attn_mask.repeat([2,1,1])

                        if args.gpt_model.startswith("R-GPT"):

                            inference_training_sample = generate_RLlamaGen(
                            model.module, t5_feat_padding, latent_size ** 2, 
                            attn_mask, 
                            cfg_scale=args.cfg_scale,
                            temperature=args.temperature, top_k=args.top_k,
                            top_p=args.top_p, sample_logits=True, 
                            vq_model = vq_model, qzshape =qzshape, 
                            image_size=args.image_size,
                            
                            )
                        else:
                            inference_training_sample = generate_with_decoder(
                            model.module, t5_feat_padding, latent_size ** 2, 
                            emb_masks=None,
                            attn_mask=attn_mask, 
                            cfg_scale=args.cfg_scale,

                            temperature=args.temperature, top_k=args.top_k,
                            top_p=args.top_p, sample_logits=True, 
                            vq_model = vq_model, qzshape =qzshape,                         
                            )

                        
                        model.module.clear_caches()
                        inference_training_sample_tb_norm = (inference_training_sample[0] + 1) / 2
                        all_visualization_images.append(inference_training_sample_tb_norm)
                        all_visualization_images_gt.append((img+1)/2) # [-1, 1] -> [0, 1]
                    
                    writer.add_image(f'Training Sample', torch.cat(all_visualization_images, dim=-1), train_steps)
                    writer.add_image(f'Training Sample GT', torch.cat(all_visualization_images_gt, dim=-1), train_steps)


                    model.train()
                dist.barrier()
                
        if not args.no_epoch_save:
            if rank == 0:
                
                model_weight = model.module.state_dict()  
                checkpoint = {
                    "model": model_weight,
                    "optimizer": optimizer.state_dict(),
                    "steps": train_steps,
                    "args": args
                }
                if not args.no_local_save:
                    checkpoint_path = f"{checkpoint_dir}/epoch_{epoch:03d}_{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    # Check and delete the oldest checkpoint if the total number exceeds the threshold
                    checkpoint_files = sorted(glob(f"{checkpoint_dir}/*.pt"), key=os.path.getctime)
                    if len(checkpoint_files) > args.max_checkpoints:
                        oldest_checkpoint = checkpoint_files[0]
                        os.remove(oldest_checkpoint)
                        logger.info(f"Deleted oldest checkpoint {oldest_checkpoint} to maintain max checkpoints limit")
                if not args.no_cloud_save:
                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/epoch_{epoch:03d}_{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")

                    # Check and delete the oldest checkpoint if the total number exceeds the threshold
                    checkpoint_files = sorted(glob(f"{cloud_checkpoint_path}/*.pt"), key=os.path.getctime)
                    if len(checkpoint_files) > args.max_checkpoints:
                        oldest_checkpoint = checkpoint_files[0]
                        os.remove(oldest_checkpoint)
                        logger.info(f"Deleted oldest checkpoint {oldest_checkpoint} to maintain max checkpoints limit")

        if exit_flag:
            break
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...


    logger.info("Done!")
    dist.destroy_process_group()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--data-paths", type=str, nargs='+')
    parser.add_argument("--t5-feat-path", type=str)

    parser.add_argument("--short-t5-feat-path", type=str, default=None, help="short caption of t5_feat_path")
    parser.add_argument("--cloud-save-path", type=str, required=False, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--no-cloud-save", action='store_true', help='no save checkpoints to cloud')

    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")

    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='t2i_cc12m')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    
    # augmentation
    parser.add_argument("--aug_remove_white_borders",  action='store_true') # aug_remove_white_borders

    # for early stop
    parser.add_argument("--early-stop", type=int, default=-1)
    parser.add_argument("--init-gpt-ckpt", type=str, default=None, help="ckpt path for init training")

    # logging
    parser.add_argument("--no_epoch_save", action='store_true', help='no_epoch_save')
    parser.add_argument("--max_checkpoints", type=int, default=5)


    # log t2i image during training
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--eval_order_version_id", type=int, default=1)

    # for validation
    parser.add_argument("--validate_every", type=int,  default=-1)
    parser.add_argument("--validation_data_path", type=str,)
    parser.add_argument("--prompt-csv", type=str, default='evaluations/t2i/coco_captions.csv')
    parser.add_argument("--num_fid_samples", type=int,  default=1600)
    parser.add_argument("--eval_clip_model4eval", default="ViT-B/32", type=str, help="[WO, ViT-B/32, ViT-G/14]")
    parser.add_argument("--eval_res", default=256, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)



    
    args = parser.parse_args()

    val_args = argparse.Namespace()
    val_args.dataset = args.dataset
    val_args.data_path = args.validation_data_path
    val_args.image_size = args.image_size
    val_args.downsample_size = args.downsample_size

    main(args, val_args)
