"""
Scaling benchmark: measures throughput at 1, 2, and 4 GPUs.
Supports both weak scaling (fixed batch per GPU) and strong scaling
(fixed total batch, split across GPUs).

Produces a CSV of results and a matplotlib plot.

Run on a multi GPU machine:
  python scaling_benchmark.py --max_gpus 4
  python scaling_benchmark.py --max_gpus 4 --scaling strong --total_batch 32

On Colab (single GPU), only the 1 GPU measurement runs:
  python scaling_benchmark.py --max_gpus 1
"""

import os
import csv
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from dist_utils import setup_distributed, cleanup_distributed, find_free_port
from models import UNet
from data_ddp import SyntheticMRIDataset, create_dataloader


NUM_SAMPLES = 400
NUM_WARMUP_BATCHES = 5
NUM_MEASURED_BATCHES = 20


def _run_training_loop(model, loader, optimizer, scaler, criterion,
                       device, n_warmup, n_measured):
    """Shared training loop for both single and multi GPU benchmarks."""
    model.train()
    batch_iter = iter(loader)

    for _ in range(n_warmup):
        try:
            inp, tgt = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            inp, tgt = next(batch_iter)
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            loss = criterion(model(inp), tgt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_samples = 0

    for _ in range(n_measured):
        try:
            inp, tgt = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            inp, tgt = next(batch_iter)
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            loss = criterion(model(inp), tgt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_samples += inp.shape[0]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, total_samples


def benchmark_worker(rank, world_size, port, batch_size, results_dict):
    setup_distributed(rank=rank, world_size=world_size,
                      backend="nccl", port=port)
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    model = UNet(channels=(32, 64, 128, 256), dropout_p=0.05).to(device)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank % torch.cuda.device_count()])

    dataset = SyntheticMRIDataset(num_samples=NUM_SAMPLES)
    loader = create_dataloader(
        dataset, batch_size=batch_size, num_workers=0,
        distributed=(world_size > 1), shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.L1Loss()

    elapsed, total_samples = _run_training_loop(
        model, loader, optimizer, scaler, criterion, device,
        NUM_WARMUP_BATCHES, NUM_MEASURED_BATCHES)

    if rank == 0:
        global_samples = total_samples * world_size
        results_dict['elapsed'] = elapsed
        results_dict['sec_per_iter'] = elapsed / NUM_MEASURED_BATCHES
        results_dict['throughput'] = global_samples / elapsed
        results_dict['total_samples'] = global_samples

    cleanup_distributed()


def run_single_gpu(batch_size=8):
    """Benchmark without DDP for the 1 GPU baseline."""
    device = torch.device("cuda:0")
    model = UNet(channels=(32, 64, 128, 256), dropout_p=0.05).to(device)
    dataset = SyntheticMRIDataset(num_samples=NUM_SAMPLES)
    loader = create_dataloader(
        dataset, batch_size=batch_size, num_workers=0,
        distributed=False, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.L1Loss()

    elapsed, total_samples = _run_training_loop(
        model, loader, optimizer, scaler, criterion, device,
        NUM_WARMUP_BATCHES, NUM_MEASURED_BATCHES)

    return {
        'elapsed': elapsed,
        'sec_per_iter': elapsed / NUM_MEASURED_BATCHES,
        'throughput': total_samples / elapsed,
        'total_samples': total_samples,
    }


def generate_plot(csv_path: str, output_path: str, scaling_mode: str = "weak"):
    """Generate scaling plot from benchmark CSV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gpus, throughputs, speedups = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gpus.append(int(row['num_gpus']))
            throughputs.append(float(row['throughput_samples_sec']))
            speedups.append(float(row['speedup']))

    if not gpus:
        print("No data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(gpus, throughputs, color='#2196F3', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Number of GPUs')
    ax1.set_ylabel('Throughput (samples/sec)')
    ax1.set_title('Training Throughput')
    ax1.set_xticks(gpus)
    for g, t in zip(gpus, throughputs):
        ax1.text(g, t + max(throughputs) * 0.02, f'{t:.1f}',
                 ha='center', fontsize=10)

    ax2.plot(gpus, speedups, 'o-', color='#F44336', linewidth=2, markersize=8)
    ax2.plot(gpus, gpus, '--', color='gray', linewidth=1, label='Ideal linear')
    ax2.set_xlabel('Number of GPUs')
    ax2.set_ylabel('Speedup vs 1 GPU')
    ax2.set_title('Scaling Efficiency')
    ax2.set_xticks(gpus)
    ax2.legend()
    for g, s in zip(gpus, speedups):
        ax2.annotate(f'{s:.2f}x', (g, s), textcoords="offset points",
                     xytext=(10, -5), fontsize=10)

    mode_label = scaling_mode.capitalize()
    fig.suptitle(f'DDP {mode_label} Scaling: UNet on fastMRI (320x320)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_gpus', type=int, default=None,
                        help='Max GPUs to benchmark. Default: all available.')
    parser.add_argument('--output_dir', type=str, default='./results/scaling')
    parser.add_argument('--scaling', type=str, default='weak',
                        choices=['weak', 'strong'],
                        help='Weak: fixed batch/GPU. Strong: fixed total batch.')
    parser.add_argument('--total_batch', type=int, default=32,
                        help='Total batch size for strong scaling mode.')
    parser.add_argument('--batch_size_per_gpu', type=int, default=8,
                        help='Batch size per GPU for weak scaling mode.')
    args = parser.parse_args()

    available = torch.cuda.device_count()
    max_gpus = args.max_gpus or available
    max_gpus = min(max_gpus, available)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'scaling_{args.scaling}.csv'
    plot_path = output_dir / f'scaling_{args.scaling}.png'

    gpu_counts = [g for g in [1, 2, 4, 8] if g <= max_gpus]

    print(f"Scaling mode: {args.scaling}")
    print(f"Available GPUs: {available}")
    print(f"Benchmarking: {gpu_counts}")
    if args.scaling == 'weak':
        print(f"Batch size per GPU: {args.batch_size_per_gpu}")
    else:
        print(f"Total batch size: {args.total_batch}")
    print(f"Warmup batches: {NUM_WARMUP_BATCHES}")
    print(f"Measured batches: {NUM_MEASURED_BATCHES}\n")

    results = []
    baseline_throughput = None

    for n_gpus in gpu_counts:
        if args.scaling == 'weak':
            bs = args.batch_size_per_gpu
            effective_bs = bs * n_gpus
        else:
            bs = max(args.total_batch // n_gpus, 1)
            effective_bs = bs * n_gpus

        print(f"--- {n_gpus} GPU(s), batch/GPU={bs}, effective={effective_bs} ---")

        if n_gpus == 1:
            r = run_single_gpu(batch_size=bs)
        else:
            manager = mp.Manager()
            r = manager.dict()
            port = find_free_port()
            mp.spawn(benchmark_worker, args=(n_gpus, port, bs, r),
                     nprocs=n_gpus, join=True)
            r = dict(r)

        if baseline_throughput is None:
            baseline_throughput = r['throughput']

        speedup = r['throughput'] / baseline_throughput
        efficiency = speedup / n_gpus

        print(f"  Throughput: {r['throughput']:.1f} samples/sec")
        print(f"  Sec/iter:   {r['sec_per_iter']:.4f}")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  Efficiency: {efficiency:.1%}\n")

        results.append({
            'num_gpus': n_gpus,
            'scaling_mode': args.scaling,
            'batch_size_per_gpu': bs,
            'effective_batch_size': effective_bs,
            'throughput_samples_sec': round(r['throughput'], 2),
            'sec_per_iter': round(r['sec_per_iter'], 5),
            'speedup': round(speedup, 3),
            'efficiency': round(efficiency, 4),
        })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

    generate_plot(str(csv_path), str(plot_path), args.scaling)


if __name__ == "__main__":
    main()
