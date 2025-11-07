from dataset.t2i import build_t2i_laion_coco, build_t2i_cc12m

# editing dataset
from dataset.t2i_editing import build_ultraedit

def build_dataset(args, **kwargs):
    # for text-to-image generation
    if args.dataset == 't2i_laion_coco':
        return build_t2i_laion_coco(args, **kwargs)
    if args.dataset  == 't2i_cc12m':
        return build_t2i_cc12m(args, **kwargs)

    # for editing
    if args.dataset == 'editing_ultraedit':
        return build_ultraedit(args, **kwargs)
    raise ValueError(f'dataset {args.dataset} is not supported')
