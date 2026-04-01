import torch
import dist_utils

def run_test(rank, world_size, port):
    dist_utils.setup_distributed(rank=rank, world_size=world_size,
                                 backend="gloo", port=port)
    print(f"  rank={dist_utils.get_rank()}, world_size={dist_utils.get_world_size()}, "
          f"main={dist_utils.is_main_process()}", flush=True)

    local_preds = torch.randn(5 + rank, 3)
    gathered = dist_utils.gather_predictions(local_preds, world_size)
    if dist_utils.is_main_process():
        print(f"  Gathered shape: {gathered.shape} (expected [11, 3])", flush=True)

    dist_utils.cleanup_distributed()
