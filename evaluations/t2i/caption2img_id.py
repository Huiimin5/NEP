# Modified from:
#   GigaGAN: https://github.com/mingukkang/GigaGAN
import os
import torch
import numpy as np
import re
import io
import random

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from PIL import Image
import torchvision.transforms as transforms
import glob

import pandas as pd
import json



def save_coco_image_ids(args):
    ''' output image : tensor to PIL
    '''
    # extract all annotation: image id mappings
    cocoval_annotation_data = json.load(open(args.cocoval_annotation_path))['annotations']
    caption_image_id_from_cocoval = {}
    total_num = 0
    for line in cocoval_annotation_data:
        if  line["caption"] in caption_image_id_from_cocoval:
            caption, image_id = line["caption"], line['image_id']
            print(f"{caption}_id_{caption_image_id_from_cocoval[caption]}_to_{image_id}")
            total_num += 1
        caption_image_id_from_cocoval[line["caption"].strip()] = line["image_id"] # save all image ids 
    print(f"A total of {total_num} captions correpond to multiple images")

    # find all matched image ids of given captions and log them into the output file

    # write_file = open(args.output_path, 'w')
    # val_caption_data = pd.read_csv(args.caption_only_path, header=None)
    val_caption_file = open(args.caption_only_path)
    id = 0
    with open(args.output_path, 'w') as f:
        
        for caption in val_caption_file: 
            if id == 0:
                f.write(f"image_ids:\n")
                id += 1
                continue
            caption = caption.strip()
            f.write(f"{caption_image_id_from_cocoval[caption]}\n")
            id += 1
    print(f"finished writing to {args.output_path}") 


        



if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_only_path", required=True, default="coco_captions.csv", help="caption_only_path")
    parser.add_argument("--cocoval_annotation_path",  required=True, default="/home/wuhuimin/data/coco/images/annotations/captions_val2014.json", help="location of the reference images for evaluation")
    parser.add_argument("--output_path", required=True, default="coco_image_ids.txt", help="output_path")

    
    # parser.add_argument("--ref_data", default="coco2014", type=str, help="in [imagenet2012, coco2014, laion4k]")
    # parser.add_argument("--ref_type", default="train/valid/test", help="Type of reference dataset")

    # parser.add_argument("--how_many", default=30000, type=int)
    # parser.add_argument("--clip_model4eval", default="ViT-B/32", type=str, help="[WO, ViT-B/32, ViT-G/14]")
    # parser.add_argument("--eval_res", default=256, type=int)
    # parser.add_argument("--batch_size", default=8, type=int)
    
    opt, _ = parser.parse_known_args()
    save_coco_image_ids(opt)