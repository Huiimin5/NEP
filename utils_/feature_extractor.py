import os
from PIL import Image
import requests
import torch
import copy





def load_llava_next(device):
    try:
        from llava.model.builder import load_pretrained_model
    except:
        print("Installing LLaVA-NeXT")
        os.system("pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git")
    from llava.model.builder import load_pretrained_model
    
    


    device_map = "auto"

    pretrained = "lmms-lab/llama3-llava-next-8b"
    model_name = "llava_llama3"
    
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, 
                                                                          attn_implementation = 'sdpa') # Add any other thing you want to pass in llava_model_args

    model.eval()
    model.tie_weights()

    return model, image_processor, tokenizer

def extract_llava_next(img_pil, model, image_processor, tokenizer): # img_tensor
    device = "cuda"
    device_map = "auto"
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX

    # url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    # image1 = Image.open(requests.get(url, stream=True).raw)
    # image_tensor1 = process_images([image1], image_processor, model.config)
    image_li = process_images(img_pil, image_processor, model.config)
    # image_li = [_image.to(dtype=torch.float16, device=device) for _image in image_li]
    # image_tensor = torch.stack(image_li)
    img_tensor = image_li[:, 0] # ?



    vision_tower = model.get_vision_tower()
    vision_tower.to_empty(device = device)
    # input_h, input_w = img_tensor.size(2), img_tensor.size(3)   
    # vision_tower_h, vision_tower_w = 334, 334
    # img_tensor = torch.nn.functional.interpolate(img_tensor, size=(vision_tower_h, vision_tower_w), mode='bilinear', align_corners=False)
    # vision_out = vision_tower(img_tensor)
    # image_features = vision_out["last_hidden_state"]
    image_features =model.encode_images(img_tensor)

    import pdb; pdb.set_trace()
    # vision_out = model.model.vision_tower(img_tensor)
    # image_features = vision_out["last_hidden_state"]
    print("Extracted features shape:", image_features.shape)
    

    # conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
    # question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
    # conv = copy.deepcopy(conv_templates[conv_template])
    # conv.append_message(conv.roles[0], question)
    # conv.append_message(conv.roles[1], None)
    # prompt_question = conv.get_prompt()

    
    # input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    # image_sizes = [image.size]

    # cont = model.generate(
    #     input_ids,
    #     images=image_tensor,
    #     image_sizes=image_sizes,
    #     do_sample=False,
    #     temperature=0,
    #     max_new_tokens=256,
    # )
    # text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    # print(text_outputs)
    # The image shows a ra


def load_blip2_t5(device):
    from PIL import Image
    import requests
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    model.to(device)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)


#================================================================================================

def rewrite_clip_visual_forward(model):
    original_forward = model.visual.forward

    def new_forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x[:,1:]

    model.visual.forward = new_forward.__get__(model.visual, model.visual.__class__)

def load_clip(device):
    try:
        import clip
    except:
        print("Installing CLIP")
        os.system("pip install git+https://github.com/openai/CLIP.git")
    
    # device = img_tensor.device#  "cuda" if torch.cuda.is_available() else "cpu"
    import clip
    model, preprocess = clip.load("ViT-L/14", device)
    rewrite_clip_visual_forward(model)
    return model, preprocess



def extract_clip(model, preprocess, img_pil_li, device):
    
    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    image_li = [preprocess(img_pil) for img_pil in img_pil_li]
    image_batch = torch.stack(image_li).to(device)


    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        
        return image_features
        # text_features = model.encode_text(text)
        
        # logits_per_image, logits_per_text = model(image, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

#================================================================================================
# load DINOv2
def forward(self, *args, is_training=False, **kwargs):
    ret = self.forward_features(*args, **kwargs)
    if is_training:
        return ret
    else:
        # return self.head(ret["x_norm_clstoken"])
        return ret["x_norm_patchtokens"]


    
def load_dino_v2(device):
    dino_v2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
    dino_v2_model.forward = forward.__get__(dino_v2_model, dino_v2_model.__class__)

    dino_v2_model.eval()

    return dino_v2_model

def extract_dino_v2(dino_v2_model, image_tensors):
    features = dino_v2_model(image_tensors)
    return features