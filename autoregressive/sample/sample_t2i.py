import torch
from utils_ import data
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image
import numpy as np
import os
import time
from language.t5 import T5Embedder
from autoregressive.models.generate import generate_magicbrush_by_next_editing_token_prediction
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json



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

