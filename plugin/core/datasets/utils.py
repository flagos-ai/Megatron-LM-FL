import os
import torch

def is_built_on_zero_rank():
    """
    Determines if the current distributed rank is the one responsible for building datasets.

    Returns:
        bool: True if the current rank is responsible for building resources, False otherwise.
    """
    
    from megatron.training import get_args
    #TODO: We should not depend on get_args in megatron core, the args belong to training.
    try: ### for unit tests
        args = get_args()
    except Exception:
        if torch.distributed.get_rank() == 0 or int(os.environ["LOCAL_RANK"]) == 0:
            return True
        else:
            return False

    is_built = False
    if not args.no_shared_fs \
        and torch.distributed.get_rank() == 0:
        is_built = True
    elif args.no_shared_fs \
        and int(os.environ["LOCAL_RANK"]) == 0:
        is_built = True
    else:
        is_built = False

    return is_built
