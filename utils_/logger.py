import logging
import torch.distributed as dist

def create_logger(logging_dir, rank):
    """
    Create a logger that writes to a log file and stdout.
    """
    # if dist.get_rank() == 0:  # real logger
    if rank ==0: # rank0 for distributed training or non-distributed training 
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger