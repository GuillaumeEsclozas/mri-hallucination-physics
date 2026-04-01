"""Minimal torchrun benchmark."""
import os, time, torch
import torch.nn as nn, torch.optim as optim, torch.distributed as dist
from dist_utils import setup_distributed, cleanup_distributed, get_rank, get_world_size
from models import UNet
from data_ddp import SyntheticMRIDataset, create_dataloader

def main():
    setup_distributed()
    rank = get_rank()
    ws = get_world_size()
    dev = torch.device(f"cuda:{int(os.environ.get(chr(76)+chr(79)+chr(67)+chr(65)+chr(76)+chr(95)+chr(82)+chr(65)+chr(78)+chr(75), 0))}")
    model = UNet(channels=(32, 64, 128, 256), dropout_p=0.05).to(dev)
    if ws > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
    ds = SyntheticMRIDataset(num_samples=800)
    loader = create_dataloader(ds, batch_size=8, num_workers=2, distributed=(ws > 1), shuffle=True, drop_last=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.L1Loss()
    model.train()
    it = iter(loader)
    for _ in range(10):
        try: inp, tgt = next(it)
        except StopIteration: it = iter(loader); inp, tgt = next(it)
        inp, tgt = inp.to(dev), tgt.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"): loss = crit(model(inp), tgt)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if dist.is_initialized(): dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total = 0
    for _ in range(30):
        try: inp, tgt = next(it)
        except StopIteration: it = iter(loader); inp, tgt = next(it)
        inp, tgt = inp.to(dev), tgt.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"): loss = crit(model(inp), tgt)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        total += inp.shape[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if rank == 0:
        gs = total * ws
        print(f"GPUs={ws} | Throughput={gs/elapsed:.1f} samples/sec | Sec/iter={elapsed/30:.4f}")
    cleanup_distributed()

if __name__ == "__main__":
    main()
