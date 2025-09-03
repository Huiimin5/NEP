from dataset.imagenet import build_imagenet, build_imagenet_code
from dataset.coco import build_coco
from dataset.openimage import build_openimage
from dataset.pexels import build_pexels
from dataset.t2i import build_t2i, build_t2i_code, build_t2i_image
from dataset.t2i import build_t2i_laion_coco, build_t2i_cc12m

# editing dataset
from dataset.t2i_editing import build_ultraedit
from dataset.t2i_editing import build_ultraedit_mask

def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    if args.dataset == 'pexels':
        return build_pexels(args, **kwargs)
    if args.dataset == 't2i_image':
        return build_t2i_image(args, **kwargs)
    if args.dataset == 't2i':
        return build_t2i(args, **kwargs)
    if args.dataset == 't2i_code':
        return build_t2i_code(args, **kwargs)
    

    # for text-to-image generation
    if args.dataset == 't2i_laion_coco':
        return build_t2i_laion_coco(args, **kwargs)
    if args.dataset  == 't2i_cc12m':
        return build_t2i_cc12m(args, **kwargs)

    # for editing
    if args.dataset == 'editing_ultraedit':
        return build_ultraedit(args, **kwargs)
    raise ValueError(f'dataset {args.dataset} is not supported')

def build_helper_dataset(args, **kwargs):
    if args.helper_dataset == 'ultraedit_editing_mask': # masks from ultraedit
        return build_ultraedit_mask(args, **kwargs)
    raise ValueError(f'dataset {args.helper_dataset} is not supported')