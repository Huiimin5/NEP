import torch
from utils_ import data
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image
import random
import numpy as np
import os
import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate

from torchvision import transforms
from dataset.build import build_dataset

from autoregressive.models.generate import generate_magicbrush_by_next_editing_token_prediction
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        attention_version = args.attention_version,
        attention_controller_mode = args.attention_controller_mode, 
        text_length = args.cls_token_num,
        teacher_forcing_length = args.num_gt_token,
        replace_attn_stop = args.replace_attn_stop,

        eval_order_version_id = args.eval_order_version_id,
        rope_correction = args.rope_correction,

        enforce_inpainting = args.enforce_inpainting,

        with_rag_image = args.with_rag_image,
        mask_in_context = args.mask_in_context,

    ).to(device=device, dtype=precision)

    print(f"GPT Parameters: {sum(p.numel() for p in gpt_model.parameters()):,}")

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    msg = gpt_model.load_state_dict(model_weight, strict=False)
    print(msg)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )



   
    # prompts = [
    #     "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grassin front of the Sydney Opera House holding a sign on the chest that says Welcome Friends!",
    #     "A blue Porsche 356 parked in front of a yellow brick wall.",
    #     "A photo of an astronaut riding a horse in the forest. There is a river in front of them with water lilies.",
    #     "A map of the United States made out of sushi. It is on a table next to a glass of red wine.",
    #     "A cute cat, hanging a card that only says OmniGen,quality details,hyper realistic,high definition.",
    #     "A cute cat, hanging a card that only says OmniGen",

    #     "The ground is covered with moss and white flowers.",
    #     "The front entrance to a restaurant with a sign that says Eat the Foodary.",
    #     "The centerpiece is made up of white and pink flowers.",
    #     "A large house with two garages and trees.",
    #     "The van is parked in front of some cotton boxes.",

    #     "The silver Range Rover is parked in the garage.",
    #     "There is an apple and some jam in the jar.",
    #     "A glass filled with pink liquid and garnished with mint.",
    #     "The road is empty and there are clouds in the sky.",        
    #     "An airplane is parked on the runway near some trees.",

    #     "A cute cat, hanging a card that only says OmniGen",


    # ]

    # prompts = prompts[args.sample_id:args.sample_id + 1]
    # caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

    
    # if not args.no_left_padding:
    #     print(f"processing left-padding...")    
    #     # a naive way to implement left-padding
    #     new_emb_masks = torch.flip(emb_masks, dims=[-1])
    #     new_caption_embs = []
    #     for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
    #         valid_num = int(emb_mask.sum().item())
    #         print(f'  prompt {idx} token len: {valid_num}')
    #         new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
    #         new_caption_embs.append(new_caption_emb)
    #     new_caption_embs = torch.stack(new_caption_embs)
    # else:
    #     new_caption_embs, new_emb_masks = caption_embs, emb_masks
    # c_indices = new_caption_embs * new_emb_masks[:,:, None]
    # c_emb_masks = new_emb_masks

    # qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
    # t1 = time.time()

    # if args.generation_version == 1:
    #     index_sample = generate(
    #         gpt_model, c_indices, latent_size ** 2, 
    #         c_emb_masks, 
    #         cfg_scale=args.cfg_scale,
    #         temperature=args.temperature, top_k=args.top_k,
    #         top_p=args.top_p, sample_logits=True, 
    #         )
    



    # inference on magicbrush test set and organize the edited images
    # elif args.generation_version == 22: 
    sample_magicbrush_test(args, gpt_model, t5_model, vq_model, latent_size, device,
    mask_in_context = args.mask_in_context,             )


def sample_magicbrush_test(args, gpt_model, t5_model, vq_model, latent_size, device, 
                           mask_in_context = False, ):
    from tqdm import tqdm
    with open(args.magicbrush_json) as fp:
        data_json = json.load(fp)
    if not os.path.exists(args.magicbrush_output_dir):
        os.makedirs(args.magicbrush_output_dir, exist_ok=True)

    def edit_image(input_path, output_path, instruction_or_caption, edit_mask_region, enforce_validation_fake_masking, with_t5_prompt, device,                 
                   mask_in_context = False,
                   ):
        # instructions are given in str
        if not with_t5_prompt:
            prompts = [instruction_or_caption]
            caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

        
            if not args.no_left_padding:
                print(f"processing left-padding...")    
                # a naive way to implement left-padding
                new_emb_masks = torch.flip(emb_masks, dims=[-1])
                new_caption_embs = []
                for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                    valid_num = int(emb_mask.sum().item())
                    print(f'  prompt {idx} token len: {valid_num}')
                    new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                    new_caption_embs.append(new_caption_emb)
                new_caption_embs = torch.stack(new_caption_embs)
            else:
                new_caption_embs, new_emb_masks = caption_embs, emb_masks
            c_indices = new_caption_embs * new_emb_masks[:,:, None]
            c_emb_masks = new_emb_masks

        # instructions are t5 features
        else:
            # load t5 features from json
            c_indices, c_emb_masks = edit_instruction            
            c_indices, c_emb_masks = c_indices.to(device), c_emb_masks.to(device)

        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        
        # generate edited image and save to  
        t1 = time.time()
        generate_magicbrush_by_next_editing_token_prediction(
            gpt_model, c_indices, latent_size ** 2, c_emb_masks, 
            source_img_path = input_path, edit_region_mask_path = edit_mask_region, enforce_validation_fake_masking = enforce_validation_fake_masking,
            save_output_img_path = output_path,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            vq_model = vq_model, qzshape =qzshape, 
            image_size=args.image_size, patch_size=args.downsample_size,
            mask_in_context = mask_in_context,
            )
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
        return sampling_time


    # iterative edit
    if args.magicbrush_skip_iter:
        print("Skip Iterative (Mutli-Turn) Editing......")
    else:
        print("Iterative (Mutli-Turn) Editing......")
        # for idx, datas in enumerate(data_json):
        for image_id, datas in tqdm(data_json.items()):
            for turn_id, data in enumerate(datas):

                image_name = data['input'] # 139306-input.png                
                image_dir = image_name.split('-')[0]

                
                # image_dir is 139306
                # 1. source image
                if turn_id == 0:  # first enter
                    image_path = os.path.join(args.magicbrush_input_path, image_dir, image_name)
                else:
                    image_path = save_output_img_path
                # 2. instruction
                # by default, use string instructions load from official magicbrush test st
                if not args.with_t5_prompt: 
                    if args.instruction_based:
                        edit_instruction = data['instruction'] # Put a cat on the seat.
                    else:
                        output_image_name = data['output'] # 139306-output1.png
                        # global_descriptions.json
                        with open(os.path.join(args.magicbrush_input_path, '..', 'global_descriptions.json')) as fp:
                            global_description_json = json.load(fp)
                        edit_instruction = global_description_json[image_dir][output_image_name] # A cat lounges on a seat under a mirror onboard a train, next to a cluttered counter
                    # 242679-output1.png
                # read the instruction as t5 embeddings files
                else:
                    t5_embedding_path = data['instruction']

                    t5_feat_padding = torch.zeros((1, args.t5_feature_max_len, args.t5_feature_dim))
                    if os.path.isfile(t5_embedding_path):
                        t5_feat = torch.from_numpy(np.load(t5_embedding_path))
                        t5_feat_len = t5_feat.shape[1] 
                        feat_len = min(args.t5_feature_max_len, t5_feat_len)
                        t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len] # [1,120,2048]
                        emb_mask = torch.zeros((args.t5_feature_max_len,))
                        emb_mask[-feat_len:] = 1
                        emb_mask = emb_mask.unsqueeze(0) # [1,120]
                    else:
                        raise ValueError(f"t5_embedding_path {t5_embedding_path} not exists")
                    edit_instruction = [t5_feat_padding, emb_mask]


                # 3. edit region mask
                edit_region_mask_name = data['mask'] # 242679-mask1.png
                edit_region_mask_name_white = edit_region_mask_name.split('.png')[0] + '_white.png'
                edit_region_mask_path = os.path.join(args.magicbrush_input_path, image_dir, edit_region_mask_name_white)

                # 4. load
                save_output_dir_path = os.path.join(args.magicbrush_output_dir, image_dir)
                if not os.path.exists(save_output_dir_path):
                    os.makedirs(save_output_dir_path)
                if turn_id == 0:
                    save_output_img_path = os.path.join(save_output_dir_path, image_dir+'_1.png')
                else:
                    save_output_img_path = os.path.join(save_output_dir_path, image_dir+'_iter_' +str(turn_id + 1)+'.png')
                
                if os.path.exists(save_output_img_path):
                    print('already generated, skip')
                    continue
                
                edit_image(image_path, save_output_img_path, edit_instruction, edit_region_mask_path,
                           enforce_validation_fake_masking = args.enforce_validation_fake_masking, 
                           with_t5_prompt = args.with_t5_prompt, device=device, 
                           mask_in_context = mask_in_context
                           )

    print("Independent (Single-Turn) Editing......")
    sampling_time_all = []
    # for idx, datas in enumerate(data_json):
    for image_id, datas in tqdm(data_json.items()):
        for turn_id, data in enumerate(datas):
            # 1. source image
            image_name = data['input']
            image_dir = image_name.split('-')[0]
            image_path = os.path.join(args.magicbrush_input_path, image_dir, image_name)


            # 2. instruction
            if not args.with_t5_prompt: 
                if args.instruction_based:
                    edit_instruction = data['instruction'] # Put a cat on the seat.
                else:
                    output_image_name = data['output'] # 139306-output1.png
                    
                    # global_descriptions.json
                    with open(os.path.join(args.magicbrush_input_path, '..', 'global_descriptions.json')) as fp:
                        global_description_json = json.load(fp)
                    edit_instruction = global_description_json[image_dir][output_image_name] # A cat lounges on a seat under a mirror onboard a train, next to a cluttered counter
            else: # using t5 embeddings instead of raw text
                t5_embedding_path = data['instruction']

                t5_feat_padding = torch.zeros((1, args.t5_feature_max_len, args.t5_feature_dim))
                if os.path.isfile(t5_embedding_path):
                    t5_feat = torch.from_numpy(np.load(t5_embedding_path))
                    t5_feat_len = t5_feat.shape[1] 
                    feat_len = min(args.t5_feature_max_len, t5_feat_len)
                    t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len] # [1,120,2048]
                    emb_mask = torch.zeros((args.t5_feature_max_len,))
                    emb_mask[-feat_len:] = 1
                    emb_mask = emb_mask.unsqueeze(0) # [1,120]
                else:
                    raise ValueError(f"t5_embedding_path {t5_embedding_path} not exists")
                edit_instruction = [t5_feat_padding, emb_mask]
            

            # 3. edit region mask
                edit_region_mask_name = data['mask'] # 242679-mask1.png
                edit_region_mask_name_white = edit_region_mask_name.split('.png')[0] + '_white.png'
                edit_region_mask_path = os.path.join(args.magicbrush_input_path, image_dir, edit_region_mask_name_white)

            save_outut_dir_path = os.path.join(args.magicbrush_output_dir, image_dir)
            if not os.path.exists(save_outut_dir_path):
                os.makedirs(save_outut_dir_path)
            if turn_id == 0:
                save_output_img_path = os.path.join(save_outut_dir_path, image_dir+'_1.png')
            else:
                save_output_img_path = os.path.join(save_outut_dir_path, image_dir+'_inde_' +str(turn_id + 1)+'.png')
            if os.path.exists(save_output_img_path):  # already generated in iterative (multi-turn) editing.
                print('already generated, skip')
                continue
            
            sampling_time = edit_image(image_path, save_output_img_path, edit_instruction, edit_region_mask_path,
                                       enforce_validation_fake_masking = args.enforce_validation_fake_masking,
                                       with_t5_prompt = args.with_t5_prompt,device=device, 
                                       mask_in_context = mask_in_context,
                       )
            sampling_time_all.append(sampling_time)
    print(f"Average sampling time for independent editing: {np.mean(sampling_time_all):.2f} seconds.")



from datasets import load_dataset
import PIL
from PIL import Image
from os.path import join
"""This is a slow implementation of the sample_emuedit_test function. T5 features for instructions are extracted on-the-fly."""
def sample_emuedit_test_slow(args, gpt_model, t5_model, vq_model, latent_size, device):
    from tqdm import tqdm

    args.resolution = args.image_size
    def convert_rgba_to_rgb_mask(rgba_image):
        """
        Convert an RGBA mask image to an RGB mask where visible areas are black (0, 0, 0) and others are white (255, 255, 255).
        
        :param rgba_image: PIL.Image - Input RGBA image.
        :return: PIL.Image - Output RGB mask.
        """
        # Ensure the image is in RGBA mode
        rgba_image = rgba_image.convert('RGBA')
        _, _, _, alpha = rgba_image.split()

        # Create a new mask image where visible areas (indicated by alpha > 0) are black, otherwise white
        binary_mask = alpha.point(lambda p: 0 if p > 0 else 255)

        # Convert the grayscale mask to RGB
        rgb_mask = PIL.Image.merge('RGB', (binary_mask, binary_mask, binary_mask))

        return rgb_mask
    def load_image(image_path, is_mask=False):
        if is_mask:
            if args.eval_no_mask:
                image = PIL.Image.new("RGB", (args.resolution, args.resolution), (255, 255, 255))
            else:
                image = PIL.Image.open(image_path).convert("RGBA")
                image = convert_rgba_to_rgb_mask(image).convert("RGB").resize((args.resolution, args.resolution))
        else:
            try:
                image = PIL.Image.open(image_path)
                image = image.convert("RGB").resize((args.resolution, args.resolution))
            except:
                import time
                time.sleep(15)
                image = PIL.Image.open(image_path)
                image = image.convert("RGB").resize((args.resolution, args.resolution))
        return image
    def collate_fn(batch):
        source_img, instruction, target_img,target_image_path, special_source_img,mask_images= [], [], [],[],[],[]

        if 'emuedit_test' in args.val_dataset:
            args.eval_no_mask = args.enforce_emu_validation_fake_masking
            for element in batch:
                source_img.append(element['image'].convert('RGB').resize((args.resolution, args.resolution)))
                target_img.append(element['image'].convert('RGB').resize((args.resolution, args.resolution)))
                instruction.append(element['instruction'])
                target_image_path.append(str(element['idx'])+".png")
                if args.mask_in_context:
                    if args.enforce_emu_validation_fake_masking:
                        mask_images.append(load_image("NONE",is_mask=True))
                    else:                      
                        raise NotImplementedError  
                        # mask_folder = args.emu_mask_folder
                        # mask_path = join(mask_folder,str(element['idx'])+"_mask.png")
                        # mask_images.append(load_image(mask_path))

        return source_img, instruction, target_img,target_image_path,special_source_img,mask_images
    data_files= {"validation": "validation-*.parquet",'test':'test-*.parquet'}
    print("args.emu_data_file")
    print(args.emu_data_file)
    dataset = load_dataset(
        args.load_data_type,
        data_files=data_files,
        data_dir=args.emu_data_file,
        split="test" 
    )
    # Check if args.valid_emuedit_test_data exists and filter the dataset
    if hasattr(args, 'valid_emuedit_test_data') and args.valid_emuedit_test_data:
        print(f"Filtering dataset using valid_emuedit_test_data: {args.valid_emuedit_test_data}")
        with open(args.valid_emuedit_test_data, 'r') as f:
            valid_ids = json.load(f)
            # Convert to just idx values since valid_ids might contain more fields
            valid_ids = [item['idx'] if isinstance(item, dict) else item for item in valid_ids]
        # Filter the dataset to only include items with idx in valid_ids
        filtered_dataset = dataset.filter(lambda example: example['idx'] in valid_ids)
        print(f"Dataset filtered: {len(dataset)} → {len(filtered_dataset)}")
        dataset = filtered_dataset
    
    sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(), shuffle=False
        )
    eval_data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.val_global_batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
        collate_fn = collate_fn
    )


    if not os.path.exists(args.emuedit_test_output_dir):
        os.makedirs(args.emuedit_test_output_dir, exist_ok=True)
    
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len)

    def edit_image(input_path, output_path, instruction_or_caption, edit_mask_region, enforce_validation_fake_masking, with_t5_prompt, device,      
                mask_in_context = False,
                   ):
        # instructions are given in str
        if not with_t5_prompt:
            prompts = [instruction_or_caption] if isinstance(instruction_or_caption, str) else instruction_or_caption
            caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

            if not args.no_left_padding:
                print(f"processing left-padding...")    
                # a naive way to implement left-padding
                new_emb_masks = torch.flip(emb_masks, dims=[-1])
                new_caption_embs = []
                for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                    valid_num = int(emb_mask.sum().item())
                    print(f'  prompt {idx} token len: {valid_num}')
                    new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                    new_caption_embs.append(new_caption_emb)
                new_caption_embs = torch.stack(new_caption_embs)
            else:
                new_caption_embs, new_emb_masks = caption_embs, emb_masks
            c_indices = new_caption_embs * new_emb_masks[:,:, None]
            c_emb_masks = new_emb_masks

        else:
            raise NotImplementedError

        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        
        # generate edited image and save to  
        generate_magicbrush_by_next_editing_token_prediction(
            gpt_model, c_indices, latent_size ** 2, c_emb_masks, 
            source_img_path = input_path, edit_region_mask_path = edit_mask_region, enforce_validation_fake_masking = enforce_validation_fake_masking,
            save_output_img_path = output_path,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            vq_model = vq_model, qzshape =qzshape, 
            image_size=args.image_size, patch_size=args.downsample_size,

            mask_in_context = mask_in_context,
            )

    
    with torch.no_grad():
        for step, tumple_data in enumerate(tqdm(eval_data_loader)):
            source_img_li,instruction_li,target_img,target_image_path_li,_,mask_images = tumple_data
            save_output_img_path_li = []

            for  target in target_image_path_li:
                target_n = join(args.emuedit_test_output_dir + '/samples',target)
                folder_path ="/".join((target_n.split('/')[:-1]))
                if  not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)
                save_output_img_path_li.append(target_n)
            edit_region_mask_path, with_mask_embedding, put_edit_after_start_token = None, False, False, 
            edit_image(source_img_li, save_output_img_path_li, instruction_li, edit_region_mask_path, 
                       enforce_validation_fake_masking = args.enforce_emu_validation_fake_masking,
                    with_t5_prompt = False, device=device, 
                    mask_in_context = args.mask_in_context)

############################################################################################################
"""This a faster implementation of the sample_emuedit_test function. T5 features for instructions are pre-extracted."""
def sample_emuedit_test(args, gpt_model, t5_model, vq_model, latent_size, device, enforce_inpainting = False,
                           with_mask_embedding = False, 
                           put_edit_after_start_token = False,
                           mask_in_context = False, copy_source_pixel = False,):
    from tqdm import tqdm

    args.resolution = args.image_size
    def convert_rgba_to_rgb_mask(rgba_image):
        """
        Convert an RGBA mask image to an RGB mask where visible areas are black (0, 0, 0) and others are white (255, 255, 255).
        
        :param rgba_image: PIL.Image - Input RGBA image.
        :return: PIL.Image - Output RGB mask.
        """
        # Ensure the image is in RGBA mode
        rgba_image = rgba_image.convert('RGBA')
        _, _, _, alpha = rgba_image.split()

        # Create a new mask image where visible areas (indicated by alpha > 0) are black, otherwise white
        binary_mask = alpha.point(lambda p: 0 if p > 0 else 255)

        # Convert the grayscale mask to RGB
        rgb_mask = PIL.Image.merge('RGB', (binary_mask, binary_mask, binary_mask))

        return rgb_mask
    def load_image(image_path, is_mask=False):
        if is_mask:
            if args.eval_no_mask:
                image = PIL.Image.new("RGB", (args.resolution, args.resolution), (255, 255, 255))
            else:

                image = PIL.Image.open(image_path).convert("RGBA")
                image = convert_rgba_to_rgb_mask(image).convert("RGB").resize((args.resolution, args.resolution))
        else:
            try:
                image = PIL.Image.open(image_path)
                image = image.convert("RGB").resize((args.resolution, args.resolution))
            except:
                import time
                time.sleep(15)
                image = PIL.Image.open(image_path)
                image = image.convert("RGB").resize((args.resolution, args.resolution))
        return image
    def collate_fn(batch):
        source_img, instruction, target_img,target_image_path, special_source_img,mask_images= [], [], [],[],[],[]
        if 'magicbrush' in args.val_dataset:
            for element in batch:
                source_img.append(element['source_img'].resize((resolution,resolution)))
                instruction.append(element['instruction'])
                target_img.append(element['target_img'].resize((resolution,resolution)))
            return source_img, instruction, target_img
        elif 'magicbrush_test' in args.val_dataset:
            for element in batch:
                source_img.append(load_image(element['source_img']))
                instruction.append(element['instruction'])
                target_img.append(load_image(element['target_img']))
                target_image_path.append(element['target_image_path'])
                if args.do_mask:
                    mask_images.append(load_image(element[args.mask_column],is_mask=True))

                if 'special_source_img' in element:
                    try:
                        special_source_img.append(load_image(join(args.output_dir + '/samples',element['special_source_img'])))
                    except Exception as e:
                        print(e,'error, sleep 15s for re-load')
                        import time
                        time.sleep(15)
                        special_source_img.append(load_image(join(args.output_dir + '/samples',element['special_source_img'])))
        elif 'emuedit_test' in args.val_dataset:
            args.eval_no_mask = args.enforce_validation_fake_masking
            for element in batch:
                source_img.append(element['image'].convert('RGB').resize((args.resolution, args.resolution)))
                target_img.append(element['image'].convert('RGB').resize((args.resolution, args.resolution)))
                instruction.append(element['instruction'])
                target_image_path.append(str(element['idx'])+".png")
                if args.mask_in_context:
                    if args.enforce_validation_fake_masking:
                        mask_images.append(load_image("NONE",is_mask=True))
                    else:
                        
                        mask_folder = args.emu_mask_folder
                        mask_path = join(mask_folder,str(element['idx'])+"_mask.png")
                        mask_images.append(load_image(mask_path))

        return source_img, instruction, target_img,target_image_path,special_source_img,mask_images
    data_files= {"validation": "validation-*.parquet",'test':'test-*.parquet'}
    dataset = load_dataset(
        args.load_data_type,
        data_files=data_files,
        data_dir=args.emu_data_file,
        split="test" 
    )
    sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(), shuffle=False
        )
    eval_data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.global_batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
        collate_fn = collate_fn
    )




    with open(args.emuedit_test_json) as fp:
        data_json = json.load(fp)
    # data_json = json.loads(args.emuedit_test_json)
    if not os.path.exists(args.emuedit_test_output_dir):
        os.makedirs(args.emuedit_test_output_dir, exist_ok=True)
    

    def edit_image(input_path, output_path, instruction_or_caption, edit_mask_region, enforce_validation_fake_masking, with_t5_prompt, device, enforce_inpainting = False,
                   debug_this_sample = False, editing_mask_version = 1, enforce_validation_zero_masking = False,
                   with_mask_embedding = False, put_edit_after_start_token = False, mask_in_context = False,
                   copy_source_pixel = False):
        # instructions are given in str
        if not with_t5_prompt:
            prompts = [instruction_or_caption] if isinstance(instruction_or_caption, str) else instruction_or_caption
            caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

        
            if not args.no_left_padding:
                print(f"processing left-padding...")    
                # a naive way to implement left-padding
                new_emb_masks = torch.flip(emb_masks, dims=[-1])
                new_caption_embs = []
                for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                    valid_num = int(emb_mask.sum().item())
                    print(f'  prompt {idx} token len: {valid_num}')
                    new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                    new_caption_embs.append(new_caption_emb)
                new_caption_embs = torch.stack(new_caption_embs)
            else:
                new_caption_embs, new_emb_masks = caption_embs, emb_masks
            c_indices = new_caption_embs * new_emb_masks[:,:, None]
            c_emb_masks = new_emb_masks

        # instructions are t5 features
        else:
            # load t5 features from json
            c_indices, c_emb_masks = edit_instruction            
            c_indices, c_emb_masks = c_indices.to(device), c_emb_masks.to(device)

        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        
        # generate edited image and save to  
        generate_magicbrush_by_next_editing_token_prediction(
            gpt_model, c_indices, latent_size ** 2, c_emb_masks, 
            source_img_path = input_path, edit_region_mask_path = edit_mask_region, enforce_validation_fake_masking = enforce_validation_fake_masking,
            save_output_img_path = output_path,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            vq_model = vq_model, qzshape =qzshape, 
            image_size=args.image_size, patch_size=args.downsample_size,
            enforce_inpainting = enforce_inpainting,
            erosion_dim = args.erosion_dim,
            debug_this_sample=debug_this_sample,
            editing_mask_version = editing_mask_version,
            enforce_validation_zero_masking = enforce_validation_zero_masking,

            with_mask_embedding = with_mask_embedding,

            put_edit_after_start_token = put_edit_after_start_token,
            mask_in_context = mask_in_context,
            copy_source_pixel = copy_source_pixel,
            )

    
    with torch.no_grad():
        # pipeline.eval()
        raise NotImplementedError("The dataset has not been filtered yet")
        for datas in tqdm(data_json):
            input_image_path,instruction, instruction_t5_path,target_image_path, mask_image_path = datas['input_image'],datas['instruction'],datas['instruction_t5_path'],datas['output_image'],datas['editing_mask_path']
            image_id = os.path.basename(input_image_path).split('.')[0]
            assert args.with_t5_prompt
            t5_embedding_path = instruction_t5_path

            t5_feat_padding = torch.zeros((1, args.t5_feature_max_len, args.t5_feature_dim))
            if os.path.isfile(t5_embedding_path):
                t5_feat = torch.from_numpy(np.load(t5_embedding_path))
                t5_feat_len = t5_feat.shape[1] 
                feat_len = min(args.t5_feature_max_len, t5_feat_len)
                t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len] # [1,120,2048]
                emb_mask = torch.zeros((args.t5_feature_max_len,))
                emb_mask[-feat_len:] = 1
                emb_mask = emb_mask.unsqueeze(0) # [1,120]
            else:
                raise ValueError(f"t5_embedding_path {t5_embedding_path} not exists")
            edit_instruction = [t5_feat_padding, emb_mask]
            save_output_img_dir = os.path.join(args.emuedit_test_output_dir, 'samples')
            os.makedirs(save_output_img_dir, exist_ok=True)
            save_output_img_path = os.path.join(save_output_img_dir, os.path.basename(input_image_path))

            edit_region_mask_path, with_mask_embedding, put_edit_after_start_token = None, False, False, 
            edit_image(input_image_path, save_output_img_path, edit_instruction, edit_region_mask_path, enforce_validation_fake_masking = args.enforce_validation_fake_masking,
                    with_t5_prompt = args.with_t5_prompt, device=device, enforce_inpainting=False,# enforce_inpainting,
                    debug_this_sample = False, editing_mask_version = args.editing_mask_version,
                    enforce_validation_zero_masking = args.enforce_validation_zero_masking,
                    with_mask_embedding = with_mask_embedding,
                    put_edit_after_start_token = put_edit_after_start_token,
                    mask_in_context = args.mask_in_context,
                    copy_source_pixel = args.copy_source_pixel)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    parser.add_argument("--refine_by_replacemment", action='store_true', default=False)
    parser.add_argument("--refiment_steps", type=int, default=100, help="refiment_steps")
    parser.add_argument("--with_rejects", action='store_true', default=False)
    parser.add_argument("--conf_threshold", type=float, default=-1.0, help="conf_threshold")
    parser.add_argument("--generation_version", type=float, default=1, help="generation_version")
    
    parser.add_argument("--sample_id", type=int, default=3, help="sample_id")
    parser.add_argument("--regeneration_where", type=str, default='random')

    parser.add_argument("--reachable_token_num", type=int, default=255, help="reachable_token_num")

    parser.add_argument("--reachable_ids", type=int, nargs = '+', help="reachable_ids")
    parser.add_argument("--teacher", type=str, default='last_round')

    parser.add_argument("--overwriting_token_number", type=int, default=1, help="overwriting_token_number")
    # v8
    parser.add_argument("--cfg_eval_min", type=float, default=0, help="cfg_eval_min")
    parser.add_argument("--cfg_eval_max", type=float, default=10, help="cfg_eval_max")
    parser.add_argument("--cfg_step", type=float, default=1, help="cfg_step")
    
    # v10
    parser.add_argument("--num_gt_token", type=int, default=-1, help="num_gt_token")
    parser.add_argument("--t2i_vis_dir", type = str, default="vis_v10_512")

    # v11: editing
    parser.add_argument("--attention_version", type=int, default=1, help="attention_version")
    parser.add_argument("--attention_controller_mode", type = str, default="replacement")
    # parser.add_argument("--teacher_forcing_length", type=int, default=1, help="teacher_forcing_length")
    parser.add_argument("--v11_vis_dir", type = str, default="vis/vis_v11")
    parser.add_argument("--replace_attn_stop", type=int, default=1000000, help="replace_attn_stop")

    # v15
    parser.add_argument("--from_code_path", action='store_true', default=False)

    # v18
    parser.add_argument("--json_file", type=str, default="/home/wuhuimin/data/magic_bush_images/magicbrush_train_with_captions.jsonl")  
    
    parser.add_argument("--triplet_dir", type=str, default='/home/wuhuimin/data/magic_bush_images/train_editing_region_masking_ds4_ssim_tolerence_0.9')

    parser.add_argument("--inpainting",  action='store_true')
    parser.add_argument("--debug_wo_image_cond",  action='store_true')

    parser.add_argument("--debug",  action='store_true')
    
    # v18.1 ; v 22.1
    # evaluation dataset
    parser.add_argument("--dataset", type=str, default='editing_ultraedit')
    parser.add_argument("--data-path", type=str, default = '/home/wuhuimin/data/ultraedit/maskbased_6_26_final_append_t5_global.jsonl')
    parser.add_argument("--enforce_fake_masking", action='store_true')


    #v19
    parser.add_argument("--eval_order_version_id", type=int, default=1)
    
    # v21: mask-based order
    parser.add_argument("--v21_order_mode", type=str, default='mask')

    # v22: magicbrush test set inference
    parser.add_argument("--magicbrush_input_path", type=str, default="/home/wuhuimin/data/MagicBrush_test/test/images")
    parser.add_argument("--magicbrush_output_dir", type=str, default="editing_results/vis_v22")
    parser.add_argument("--magicbrush_skip_iter",  action='store_true')
    parser.add_argument("--instruction_based",  action='store_true') # instruction for prompt or global caption for prompt

    parser.add_argument("--enforce_validation_fake_masking", action='store_true')

    parser.add_argument("--magicbrush_json", type=str, default="/home/wuhuimin/data/MagicBrush_test/test/edit_sessions.json")

    parser.add_argument("--with_t5_prompt",  action='store_true') # instruction for prompt extracted by flan t5
    
    parser.add_argument("--rope_correction",  action='store_true') # rope does not affect the conditioning text embedding
    

    parser.add_argument("--enforce_inpainting",  action='store_true') # enforce inpainting  
    parser.add_argument("--with_rag_image",  action='store_true') # put real image containing target object in the context
    
    
    parser.add_argument("--debug_output_as_input",  action='store_true') # output image as the input
    parser.add_argument("--editing_mask_version", type=int,  default=1)
    parser.add_argument("--enforce_validation_zero_masking", action='store_true')
    parser.add_argument("--enforce_validation_copy_output", action='store_true')
    parser.add_argument("--erosion_dim", type=int,  default=0)

    parser.add_argument("--mask_in_context",  action='store_true') # [text, source image, mask] -> [target image]
    parser.add_argument("--copy_source_pixel",  action='store_true') # [text, source image, mask] -> [target image]

    # for rebuttal v22.2:  zero-shot editing with randomized order
    parser.add_argument("--randomized_orders",  action='store_true') # [text, source image, mask] -> [target image]


    args = parser.parse_args()
    main(args)
