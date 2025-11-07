import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image 
import json
import torchvision.transforms as T
from torchvision import transforms
from torch.nn import functional as F
"""
append target_caption_t5_path for each image to the orginal input jsonl file
"""



"""
======================= UltraEdit =======================
"""

class UltraEdit( Dataset): 
    def __init__(self, args, transform, return_embd_mask = False):
        self.img_path_list = []
        self.t5_feat_path_list = []
        self.target_img_path_list = []
        self.editing_mask_path_list = []

        # just in case it will be used for evaluation or visualization
        self.instructions_list = []

        with open(args.data_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                self.img_path_list.append(data['file_name'])
                self.target_img_path_list.append(data['edited_image'])
                self.editing_mask_path_list.append(data['mask_image'])
                self.t5_feat_path_list.append(data['target_caption_t5_path']) # os.path.join(args.t5_feat_path, os.path.basename(img_path).replace('.jpg', '.npy'))

                # just in case it will be used for evaluation or visualization
                self.instructions_list.append(data['edit_prompt'])

        self.transform = transform
        self.normalization_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.latent_size = latent_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        
        # for causal attention mask
        self.max_seq_length = self.t5_feature_max_len + self.code_len + self.code_len
        if args.mask_in_context:
            self.max_seq_length += self.code_len

        # raster scan
        self.raster_scan = torch.arange(self.code_len)
        self.args = args

        self.return_embd_mask = return_embd_mask

        self.mask_in_context = args.mask_in_context
    def __len__(self):
        return len(self.img_path_list)

    def dummy_data(self):
        img_source = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        img_target = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
        generation_orders_for_editing = torch.zeros((self.code_len,), dtype=torch.long)
        editing_mask = torch.zeros((self.code_len,), dtype=torch.bool)
        valid = 0
        return img_source, img_target, t5_feat_padding, generation_orders_for_editing, editing_mask,  attn_mask, torch.tensor(valid)

    def __getitem__(self, index):
        img_path_source  =  self.img_path_list[index]
        target_img_path = self.target_img_path_list[index]
        editing_mask_path = self.editing_mask_path_list[index]
        t5_file = self.t5_feat_path_list[index]
        img_source = np.array(Image.open(img_path_source).convert("RGB"))
        img_target = np.array(Image.open(target_img_path).convert("RGB"))

        if self.args.enforce_fake_masking: # all tokens are considered editing region; similar to EditAR
            img_source_h, img_source_w = img_source.shape[:2]
            mask = np.ones((img_source_h, img_source_w, 3), dtype=img_target.dtype) * 255
        else:
            try:
                mask = np.array(Image.open(editing_mask_path))[:,:,np.newaxis].repeat(3, axis=2) # [512,512,3]
            except Exception as e:
                # print(e)
                if self.args.fake_masking:
                    assert editing_mask_path[-4:]== 'NONE'
                    img_source_h, img_source_w = img_source.shape[:2]
                    mask = np.ones((img_source_h, img_source_w, 3), dtype=img_target.dtype) * 255
            
        img_triplet_before_aug = np.stack([img_source, img_target, mask], axis=0)
        img_triplet_before_aug_tensor = torch.from_numpy(img_triplet_before_aug).permute(0, 3, 1,2)# (3, 3,500, 500)
        img_triplet = self.transform(img_triplet_before_aug_tensor)
        img_triplet = img_triplet / 255.0
        img_source_rgb, img_target_rgb, mask = img_triplet.chunk(3)

        # transform for source and target
        img_source, img_target = self.normalization_transform(torch.cat([img_source_rgb, img_target_rgb])).chunk(2)
        img_source, img_target = img_source[0], img_target[0]

        # from mask to editing generation order
        mask_1c = mask[:,:1]
        # mask_ds = F.interpolate(mask * 255, size=(self.latent_size, self.latent_size), mode='nearest')
        uniform_size = 512
        
        # mask_ = F.interpolate(mask_1c * 255, size=(uniform_size, uniform_size), mode='bicubic')
        mask_ = F.interpolate((mask_1c>0.5).float(), size=(uniform_size, uniform_size), mode='bicubic')

        # Downsampling by taking the maximum of each latent_size * latent_size block
        mask_ds = F.max_pool2d(mask_, kernel_size=(uniform_size//self.latent_size,uniform_size// self.latent_size))

        mask_ds = (mask_ds > 0).int().view(-1)
        if mask_ds.sum() == 0:
            print(index, 'no editing region mask', self.editing_mask_path_list[index])
        # editing positions + unediting positions
        generation_orders_for_editing = torch.cat([self.raster_scan[mask_ds == 1], self.raster_scan[mask_ds == 0]])
        # length = 256, 1s go first and 0s next
        editing_mask = mask_ds[generation_orders_for_editing] 

        # if self.with_mask_embedding or self.mask_in_context:
        if self.mask_in_context:
            generation_orders_for_editing_unediting_first = torch.cat([self.raster_scan[mask_ds == 0], self.raster_scan[mask_ds == 1]])


        if editing_mask.sum == 0:
            img_source, img_target, t5_feat_padding, generation_orders_for_editing, editing_mask,  attn_mask, valid = \
                self.dummy_data()
            print('no editing region')

        else:
            t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
            if os.path.isfile(t5_file):
                try:
                    t5_feat = torch.from_numpy(np.load(t5_file))
                    t5_feat_len = t5_feat.shape[1] 
                    feat_len = min(self.t5_feature_max_len, t5_feat_len)
                    t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
                    emb_mask = torch.zeros((self.t5_feature_max_len,))
                    emb_mask[-feat_len:] = 1
                    
                    
                    attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
                    T = self.t5_feature_max_len
                    attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
                    eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
                    attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
                    attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
                    valid = 1
                except ValueError as e:
                    print(f"Error loading image {t5_file}: {e}")
                    img_source, img_target, t5_feat_padding, generation_orders_for_editing, editing_mask,  attn_mask, valid = self.dummy_data()
            else:
                print(f"File {t5_file} does not exist")
                img_source, img_target, t5_feat_padding, generation_orders_for_editing, editing_mask,  attn_mask, valid = self.dummy_data()

        if not self.return_embd_mask: # by default, return attention mask
            # if not self.with_mask_embedding and not self.mask_in_context: # editing by default
            if not self.mask_in_context: # editing by default
                return  img_target, img_source, t5_feat_padding, generation_orders_for_editing, editing_mask,  attn_mask, torch.tensor(valid)
            elif self.mask_in_context:
                return  img_target, img_source, t5_feat_padding, generation_orders_for_editing, generation_orders_for_editing_unediting_first, editing_mask,  attn_mask, torch.tensor(valid)
            
        else:
            return img_target, img_source, t5_feat_padding, generation_orders_for_editing, editing_mask,  emb_mask, torch.tensor(valid)

def build_ultraedit(args,transform, split = 'training', return_embd_mask = False): # split = 'training' or 'test'
    if split == 'training':
        return UltraEdit(args, transform, return_embd_mask)


