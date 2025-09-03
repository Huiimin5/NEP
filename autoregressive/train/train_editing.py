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

from utils_.distributed import init_distributed_mode
from utils_.logger import create_logger
from dataset.build import build_dataset
from autoregressive.train.train_c2i import creat_optimizer
from autoregressive.models.gpt import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models
import signal

import shutil
import numpy as np


from torch.utils.tensorboard import SummaryWriter
from autoregressive.models.generate import generate_wo_saving_by_next_editing_token_prediction
from evaluations.editing.image_eval import callable_evaluation
from evaluations.editing.emuedit_eval import callable_evaluation as callable_evaluation_emuedit
from autoregressive.sample.sample_t2i import sample_magicbrush_test, sample_emuedit_test_slow
import tempfile

# for feature alignment
from utils_.feature_extractor import load_dino_v2, extract_dino_v2


def main(args):
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
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank )
        logger.info(f"Experiment directory created at {experiment_dir}")

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
        logger = create_logger(None, -1)

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

        set_max_len = args.set_max_len,
        mask_in_context = args.mask_in_context,
        
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # Setup data:
    # if args.dataset.startswith('t2i'):     # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint        
    
    train_transforms = transforms.Compose(
        [   transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size) if args.center_crop else transforms.RandomCrop(args.image_size),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            # transforms.ToTensor(),
        ]
    )


    dataset = build_dataset(args, transform=train_transforms)

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

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / (args.global_batch_size / args.gradient_accumulation_steps)))
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

    # 
    if args.lr_scheduler_annealing == 'none':
        lr_scheduler = None
    else:

        num_update_steps_per_epoch = len(loader) // args.global_batch_size            
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, args.lr,
            epochs=args.epochs,
            steps_per_epoch = num_update_steps_per_epoch,
            pct_start=args.pct_start,
            cycle_momentum=False,
            anneal_strategy=args.lr_scheduler_annealing,
            final_div_factor = args.final_div_factor,
            last_epoch=train_steps if args.gpt_ckpt else -1
        )

    
    if args.distributed:
        model_wo_ddp = model
        model = DDP(model.to(device), device_ids=[args.gpu])
    

    def save_checkpoint_on_exit(signal, frame):
        if rank == 0:
            model_weight = model.module.state_dict()
            checkpoint = {
                "model": model_weight,
                "optimizer": optimizer.state_dict(),
                "steps": train_steps,
                "args": args
            }
            if not args.no_local_save:
                checkpoint_path = f"{checkpoint_dir}/interrupted_{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path} due to unexpected exit")
            if not args.no_cloud_save:
                cloud_checkpoint_path = f"{cloud_checkpoint_dir}/interrupted_{train_steps:07d}.pt"
                torch.save(checkpoint, cloud_checkpoint_path)
                logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path} due to unexpected exit")
        dist.destroy_process_group()
        exit(0)
    if rank == 0:
        signal.signal(signal.SIGINT, save_checkpoint_on_exit)
        signal.signal(signal.SIGTERM, save_checkpoint_on_exit)

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
    print_nvidia_every = 10000
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for data in loader:
            
            pixels_target, pixels_source, y, generation_orders_for_editing, _, editing_mask, attn_mask, valid = data            

            if args.early_stop != -1 and train_steps == args.early_stop:
                exit_flag = True
                break
            pixels_target = pixels_target.to(device, non_blocking=True)
            pixels_source = pixels_source.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            # tokenization for source image
            tokens = []
            for pixels in [pixels_target, pixels_source]:
                with torch.no_grad():
                    _, _, [_, _, indices] = vq_model.encode(pixels)
                x = indices.reshape(pixels.shape[0], -1)
                tokens.append(x)
            x, x_cond  = tokens
            

            
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(y.shape[0], y.shape[-2], y.shape[-1])
            assert z_indices.shape[0] == c_indices.shape[0]
            attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, attn_mask.shape[-2], attn_mask.shape[-1]) # (bs, n_head, seq_len, seq_len)
            try:
                if args.gpt_model.startswith("R-GPT"): # rar training
                    with torch.amp.autocast('cuda', dtype=ptdtype):  
                        logtis, loss = model(cond_idx=c_indices, idx=z_indices, targets=z_indices, mask=attn_mask[:, :, :-1,:-1], valid=valid,
                                        cond_idx_img = x_cond, generation_orders_for_editing = generation_orders_for_editing, 
                                        editing_mask = editing_mask,
                                        )

            except Exception as e:
                logger.error(f"Error during forward pass: {e}")
                raise
            # backward pass, with gradient scaling if training in fp16         
            logtis_list = torch.tensor(-1, device=device) if logtis is None else torch.tensor(0, device=device)
            dist.all_reduce(logtis_list, op=dist.ReduceOp.SUM)  # check
            if logtis_list.item() == 0:
                loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                
                if (train_steps + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if lr_scheduler:
                        lr_scheduler.step()
            else:
                logger.error(f"Error during forward pass: logtis is None")
                
            # Log loss values:
            running_loss += loss.item()
            running_loss_ce += loss.item()
            # print(running_loss)
            if train_steps % print_nvidia_every == 0:
                os.system('nvidia-smi')
            log_steps += 1
            train_steps += 1
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
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()


                # Log loss to TensorBoard
                # sampled images to Tensorboard
                if rank == 0: 
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                    writer.add_scalar('Loss_ce/train', avg_loss, train_steps) # loss is ce loss
                    if lr_scheduler:
                        writer.add_scalar('LR', lr_scheduler.get_last_lr()[0], train_steps) # loss is ce loss
                    else:
                        writer.add_scalar('LR',  args.lr, train_steps) # loss is ce loss


                    model.eval()                    
                    visualization_image_ids = [0, 4, 16, 64, 128]
                    all_visualization_images = []
                    for visualization_image_id in visualization_image_ids:
                        try:
                            data = dataset[visualization_image_id]
                        except Exception as e:
                            print(e)
                            continue

                        if not args.mask_in_context: 
                            pixels_target, pixels_source, t5_feat_padding, generation_orders_for_editing, editing_mask, attn_mask, _  = data
                            pixels_target, pixels_source, t5_feat_padding, generation_orders_for_editing, editing_mask, attn_mask = \
                            pixels_target.to(device), pixels_source.to(device), t5_feat_padding.to(device), generation_orders_for_editing.to(device), editing_mask.to(device), attn_mask.to(device)
                        
                        else:# editing with mask in context
                            pixels_target, pixels_source, t5_feat_padding, generation_orders_for_editing, _, editing_mask, attn_mask, _  = data
                            pixels_target, pixels_source, t5_feat_padding, generation_orders_for_editing, editing_mask, attn_mask = \
                            pixels_target.to(device), pixels_source.to(device), t5_feat_padding.to(device), generation_orders_for_editing.to(device), editing_mask.to(device), attn_mask.to(device)

                        sequential_mask = torch.zeros_like(editing_mask)
                        sequential_mask[generation_orders_for_editing[editing_mask==1]] = 1
                        qzshape = [1, args.codebook_embed_dim, latent_size, latent_size]
                        if args.cfg_scale != 1:
                            attn_mask = attn_mask.repeat([2,1,1])

                        if args.mask_in_context:
                            raster_scan_to_pos = generation_orders_for_editing.argsort(dim = -1).unsqueeze(0)
                            raster_scan_mask = editing_mask.unsqueeze(0).gather(1, raster_scan_to_pos)
                            if args.cfg_scale != 1:
                                raster_scan_mask = raster_scan_mask.repeat(2,1)
                        else:
                            raster_scan_mask = None

                        inference_training_sample = generate_wo_saving_by_next_editing_token_prediction(
                        model.module, t5_feat_padding, latent_size ** 2, 
                        attn_mask, source_img = pixels_source.unsqueeze(0), target_img = pixels_target.unsqueeze(0),
                        mask_resized_1d = sequential_mask.unsqueeze(0),
                        
                        cfg_scale=args.cfg_scale,
                        temperature=args.temperature, top_k=args.top_k,
                        top_p=args.top_p, sample_logits=True, 
                        vq_model = vq_model, qzshape =qzshape, 
                        image_size=args.image_size,

                        raster_scan_mask = raster_scan_mask,

                        mask_in_context = args.mask_in_context, # put mask in the AR context

                        )
                        model.module.clear_caches()
                        inference_training_sample_tb_norm = (inference_training_sample[0] + 1) / 2
                        all_visualization_images.append(inference_training_sample_tb_norm)                    
                    writer.add_image(f'Training Sample', torch.cat(all_visualization_images, dim=-2), train_steps)

                    model.train()

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

                        checkpoint_files = sorted(glob(f"{checkpoint_dir}/*.pt"), key=os.path.getctime)
                        if len(checkpoint_files) > args.max_checkpoints:
                            oldest_checkpoint = checkpoint_files[0]
                            os.remove(oldest_checkpoint)
                            logger.info(f"Deleted oldest checkpoint {oldest_checkpoint} to maintain max checkpoints limit")
                    if not args.no_cloud_save:
                        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, cloud_checkpoint_path)
                        logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

            if args.validate_every != -1 and train_steps % args.validate_every == 0:
                dist.barrier()
                model.eval()

                ckpt_string_name = f"{checkpoint_dir}-{train_steps:07d}" 

                if rank == 0:                    

                    # prepare parameters for sampling
                    # Create a temporary directory
                    if 'magicbrush' in args.val_dataset:
                        sampling_save_path = os.path.join(os.path.dirname(ckpt_string_name), f"samples-ckpt-{train_steps:07d}" )                
                        os.makedirs(sampling_save_path, exist_ok=True)
                        print('saving the sampled results to ', args.magicbrush_output_dir)
                        sample_magicbrush_test(args, model.module, None, vq_model, latent_size, loss.device, 
                                            mask_in_context = args.mask_in_context,
                                            )
                        metric_save_path = os.path.join(os.path.dirname(ckpt_string_name), f"scores-ckpt-{train_steps:07d}.txt" )
                        evaluated_metrics_dict = callable_evaluation(args.magicbrush_output_dir, args.gt_path, args.caption_path, 
                                            args.metric, metric_save_path, loss.device)
                        for turn_id, metrics in evaluated_metrics_dict.items():
                            for metric_name, metric_value in metrics.items():
                                writer.add_scalar(f'{turn_id}_{metric_name}/eval', metric_value, train_steps)

                        model.train()

                if 'emuedit_test' in args.val_dataset:
                    emuedit_sampling_save_path = os.path.join(os.path.dirname(ckpt_string_name), f"emuedit_test_samples-ckpt-{train_steps:07d}" )                
                    os.makedirs(emuedit_sampling_save_path, exist_ok=True)
                    args.emuedit_test_output_dir = emuedit_sampling_save_path
                    print('saving the sampled results to ', emuedit_sampling_save_path)
                    sample_emuedit_test_slow(args, model.module, None, vq_model, latent_size, loss.device)
                    dist.barrier() # wait for all ranks to finish sampling
                    if rank == 0: # only evaluate on one rank
                        metric_save_path = os.path.join(os.path.dirname(ckpt_string_name), f"emuedit_test_scores-ckpt-{train_steps:07d}.txt" )
                        evaluated_metrics_dict = callable_evaluation_emuedit(os.path.join(emuedit_sampling_save_path, 'samples'), args.emuedit_gt_path, args.emuedit_caption_path, 
                                            args.emuedit_metric, metric_save_path, loss.device, args.num_workers)

                    model.train()
                
            dist.barrier()
            

            
        # save checkpoint for each epoch
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
    parser.add_argument("--dataset", type=str, default='editing_ultraedit')
    parser.add_argument("--fake_masking", action='store_true')
    parser.add_argument("--enforce_fake_masking", action='store_true')
    parser.add_argument("--enforce_emu_validation_fake_masking", action='store_true')
    
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

    # for laion-coco preparation
    parser.add_argument("--prepare_dataset", action='store_true')
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    
    parser.add_argument("--early-stop", type=int, default=-1)
    parser.add_argument("--init-gpt-ckpt", type=str, default=None, help="ckpt path for init training")
    parser.add_argument("--no_epoch_save", action='store_true', help='no_epoch_save')
    parser.add_argument("--max_checkpoints", type=int, default=3)
    parser.add_argument("--set_max_len", action='store_true', help='set_max_len')

    #  for editing dataset
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )

    # tensorboard logging during training
    # log t2i image during training
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")


    # lr scheduler
    parser.add_argument(
        "--lr_scheduler_annealing",
        type=str,
        default="none",
        choices=["none", "linear", "cos", "exponential"],
    )
    parser.add_argument("--final_div_factor", type=float,  default=1e4) # -> 0.04
    parser.add_argument("--pct_start", type=float,  default=0.05) # 

    # for validation
    parser.add_argument("--validate_every", type=int,  default=-1)
    parser.add_argument("--val_dataset", type=str, default='magicbrush',) # 'magicbrush,emuedit'


    # for magicbrush test set inference
    parser.add_argument("--magicbrush_input_path", type=str, required=True, help='input image folder for magicbrush test set, eg. /home/username/data/MagicBrush_test/test/images')
    parser.add_argument("--magicbrush_skip_iter",  action='store_true', help='skip iterative (multi-turn) editing for magicbrush test set')
    parser.add_argument("--instruction_based",  action='store_true', help='whether to use instructions or global captions as prompts')
    parser.add_argument("--magicbrush_json", type=str, required=True, help='edit sessions for magicbrush test set, eg., /home/username/data/MagicBrush_test/test/edit_sessions_with_instruction_t5.json')
    parser.add_argument("--with_t5_prompt",  action='store_true', help='whether to use instructions extracted by Flan T5') 
    parser.add_argument("--enforce_validation_fake_masking", action='store_true', help='whether to use fake masks (full image as editing region) during validation')
    

    # for magicbrush test set evaluation
    parser.add_argument('--gt_path', type=str,required=True,
                        help='Paths to the gt images (folders), eg. /home/username/data/MagicBrush_test/test/images/')
    parser.add_argument('--caption_path', type=str, required=True,
                        help='the file path to store the captions for text-image similarity calculation, eg., /home/username/data/MagicBrush_test/test/local_descriptions.json')
    parser.add_argument('--metric',
                        type=str,
                        default='l1,l2,clip-i,dino,clip-t',
                        help='the metric to calculate (l1, l2, clip-i, dino, clip-t)')

    # for magicbrush data loader
    parser.add_argument("--t5_feature_max_len", type=int,  default=120)
    parser.add_argument("--t5_feature_dim", type=int,  default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)


    # for emuedit validation
    parser.add_argument(
        "--emu_data_file",
        type=str,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
            " eg. /home/username/data/emu_edit_test_set/data"
        ),
    )
    parser.add_argument(
        "--load_data_type",
        type=str,
        default="parquet",
        help="The type of the dataset to load.",
    )
    parser.add_argument(
        "--emuedit_test_json",
        type=str,
        help="json file for emuedit test set, eg./home/username/data/emu_edit_test_set/emuedit_test_append_t5_instructions.jsonl",
    )
    parser.add_argument(
        "--valid_emuedit_test_data",
        type=str,
        help="json file for emuedit test set, eg. /home/username/data/emu_edit_test_set/emu_test_data.json",
    )
    parser.add_argument('--emuedit_gt_path',
                        type=str,
                        help='Paths to the gt images (folders), eg. /home/username/data/emu_edit_test_set/images_valid/')
    parser.add_argument('--emuedit_caption_path',
                        type=str,
                        help='the file path to store the global captions for text-image similarity calculation, eg., /home/username/data/emu_edit_test_set/emu_test_data.json')
    parser.add_argument('--emuedit_metric',
                        type=str,
                        default='l1,clip-i,dino,clip-t',
                        help='the metric to calculate (l1, clip-i, dino, clip-t)')
    parser.add_argument('--val-global-batch-size', type=int, default=1)


    parser.add_argument("--mask_in_context",  action='store_true', help='if true,[text, source image, mask] -> [target image]')     
    
    
    args = parser.parse_args()
    main(args)
