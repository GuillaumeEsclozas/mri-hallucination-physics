"""
Distributed training script for MRI reconstruction U-Net.
Supports single GPU, torchrun multi GPU, and mp.spawn testing.
Configs managed via Hydra compose API for torchrun compatibility.
"""

import os
import sys
import time
import math
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from pathlib import Path
from torch.profiler import (
    profile, ProfilerActivity, schedule, tensorboard_trace_handler,
)

from dist_utils import (
    setup_distributed, cleanup_distributed,
    get_rank, get_local_rank, get_world_size, is_main_process,
)
from data_ddp import FastMRISliceDataset, SyntheticMRIDataset, create_dataloader
from models import UNet


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                    best_val_loss, train_loss, val_loss, checkpoint_dir,
                    is_best, distributed):
    model_state = model.module.state_dict() if distributed else model.state_dict()
    state = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / 'checkpoint_latest.pt')
    if is_best:
        torch.save(state, checkpoint_dir / 'checkpoint_best.pt')
        print(f"  New best model (val_loss={val_loss:.4f})")


def validate(model, val_loader, criterion, device, amp_enabled):
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for input_img, target_img in val_loader:
            input_img = input_img.to(device, non_blocking=True)
            target_img = target_img.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                loss = criterion(model(input_img), target_img)
            loss_sum += loss.item()
            n_batches += 1
    val_loss = loss_sum / max(n_batches, 1)
    if dist.is_initialized():
        loss_tensor = torch.tensor([val_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = loss_tensor.item() / get_world_size()
    return val_loss


def run_profiled_epoch(model, train_loader, optimizer, scaler, criterion,
                       device, amp_enabled, profile_dir, max_grad_norm):
    prof_dir = Path(profile_dir)
    prof_dir.mkdir(parents=True, exist_ok=True)
    prof_schedule = schedule(wait=1, warmup=2, active=10, repeat=1)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities, schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(str(prof_dir)),
        record_shapes=True, profile_memory=True, with_stack=True,
    ) as prof:
        model.train()
        for batch_idx, (input_img, target_img) in enumerate(train_loader):
            if batch_idx >= 13:
                break
            input_img = input_img.to(device, non_blocking=True)
            target_img = target_img.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                pred = model(input_img)
                loss = criterion(pred, target_img)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            prof.step()
    if is_main_process():
        print("\n--- Profiler Summary (top 15 CUDA ops by total time) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        print(f"\nTrace saved to: {prof_dir}")


def train(data_dir=None, val_dir=None, checkpoint_dir='./checkpoints',
          resume_from=None, epochs=50, batch_size=8, lr=1e-3,
          num_workers=2, acceleration=4, center_fraction=0.08,
          mask_type='random', max_train_volumes=None, max_val_volumes=None,
          checkpoint_every=10, seed=42, deterministic=False,
          skip_nan_batches=False, synthetic=False,
          model_channels=(32, 64, 128, 256), model_dropout_p=0.05,
          do_profile=False, profile_dir='./profile_traces',
          max_grad_norm=1.0, warmup_epochs=0,
          use_wandb=False, wandb_project='mri-hallucination-physics',
          rank=None, world_size_arg=None, port=None):
    setup_distributed(rank=rank, world_size=world_size_arg, port=port)
    try:
        _train_impl(
            data_dir=data_dir, val_dir=val_dir,
            checkpoint_dir=checkpoint_dir, resume_from=resume_from,
            epochs=epochs, batch_size=batch_size, lr=lr,
            num_workers=num_workers, acceleration=acceleration,
            center_fraction=center_fraction, mask_type=mask_type,
            max_train_volumes=max_train_volumes,
            max_val_volumes=max_val_volumes,
            checkpoint_every=checkpoint_every, seed=seed,
            deterministic=deterministic,
            skip_nan_batches=skip_nan_batches, synthetic=synthetic,
            model_channels=model_channels, model_dropout_p=model_dropout_p,
            do_profile=do_profile, profile_dir=profile_dir,
            max_grad_norm=max_grad_norm, warmup_epochs=warmup_epochs,
            use_wandb=use_wandb, wandb_project=wandb_project,
        )
    finally:
        cleanup_distributed()


def _train_impl(data_dir, val_dir, checkpoint_dir, resume_from,
                epochs, batch_size, lr, num_workers, acceleration,
                center_fraction, mask_type, max_train_volumes,
                max_val_volumes, checkpoint_every, seed, deterministic,
                skip_nan_batches, synthetic, model_channels, model_dropout_p,
                do_profile, profile_dir, max_grad_norm, warmup_epochs,
                use_wandb, wandb_project):
    distributed = get_world_size() > 1
    local_rank = get_local_rank()
    world_size = get_world_size()
    warmup_epochs = warmup_epochs or 0

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    amp_enabled = device.type == 'cuda'

    set_seed(seed, deterministic)

    if is_main_process():
        print(f"Training: epochs={epochs}, batch_size={batch_size}, "
              f"lr={lr}, world_size={world_size}")
        print(f"Device: {device}, AMP: {amp_enabled}, Distributed: {distributed}")
        if max_grad_norm is not None:
            print(f"Gradient clipping: max_norm={max_grad_norm}")
        if warmup_epochs > 0:
            print(f"LR warmup: {warmup_epochs} epochs")

    # ---- Data ----
    if synthetic:
        train_ds = SyntheticMRIDataset(num_samples=200, acceleration=acceleration)
        val_ds = SyntheticMRIDataset(num_samples=50, acceleration=acceleration)
        if is_main_process():
            print(f"Synthetic data: train={len(train_ds)}, val={len(val_ds)}")
    else:
        if data_dir is None or val_dir is None:
            raise ValueError("data_dir and val_dir required unless synthetic=True")
        train_ds = FastMRISliceDataset(
            data_dir, acceleration=acceleration,
            center_fraction=center_fraction, mask_type=mask_type,
            target_type='reconstruction_esc', fixed_masks=False,
            max_volumes=max_train_volumes, augment=True,
        )
        val_ds = FastMRISliceDataset(
            val_dir, acceleration=acceleration,
            center_fraction=center_fraction, mask_type=mask_type,
            target_type='reconstruction_esc', fixed_masks=True,
            max_volumes=max_val_volumes, augment=False,
        )

    train_loader = create_dataloader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        distributed=distributed, shuffle=True, drop_last=True,
    )
    val_loader = create_dataloader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        distributed=distributed, shuffle=False, drop_last=False,
    )

    # ---- Model ----
    model = UNet(channels=tuple(model_channels), dropout_p=model_dropout_p)
    start_epoch = 0
    best_val_loss = float('inf')
    ckpt = None
    if resume_from is not None:
        if is_main_process():
            print(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ---- Optimizer, Scheduler, Scaler ----
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if warmup_epochs > 0 and warmup_epochs < epochs:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0,
            total_iters=warmup_epochs)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5)

    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    criterion = nn.L1Loss()

    if ckpt is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        if is_main_process():
            print(f"Resumed: epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ---- Profile mode ----
    if do_profile:
        if is_main_process():
            print("Profiling mode: running 13 batches\n")
        run_profiled_epoch(model, train_loader, optimizer, scaler, criterion,
                           device, amp_enabled, profile_dir, max_grad_norm)
        return

    # ---- wandb ----
    wandb_run = None
    if use_wandb and is_main_process():
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                config={'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
                        'acceleration': acceleration, 'seed': seed,
                        'world_size': world_size})
        except ImportError:
            print("wandb not installed, skipping")

    # ---- JSON log setup ----
    log_path = Path(checkpoint_dir) / 'train_log.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        effective_bs = batch_size * world_size
        print(f"Effective batch size: {effective_bs} "
              f"({batch_size} x {world_size} GPUs)\n")

    # ---- Training Loop ----
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        if distributed and hasattr(train_loader, 'sampler'):
            train_loader.sampler.set_epoch(epoch)

        train_loss_sum = 0.0
        n_batches = 0
        nan_count = 0

        for input_img, target_img in train_loader:
            input_img = input_img.to(device, non_blocking=True)
            target_img = target_img.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                pred = model(input_img)
                loss = criterion(pred, target_img)
            loss_val = loss.item()

            # Always call backward to avoid DDP deadlock.
            # GradScaler handles inf/nan from AMP automatically.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if math.isfinite(loss_val):
                train_loss_sum += loss_val
                n_batches += 1
            elif skip_nan_batches:
                nan_count += 1
            else:
                train_loss_sum += loss_val
                n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        if distributed:
            tl = torch.tensor([train_loss], device=device)
            dist.all_reduce(tl, op=dist.ReduceOp.SUM)
            train_loss = tl.item() / world_size

        val_loss = validate(model, val_loader, criterion, device, amp_enabled)
        scheduler.step()
        elapsed = time.time() - t0

        if is_main_process():
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            lr_now = scheduler.get_last_lr()[0]
            status = (f"Epoch {epoch+1}/{epochs} | "
                      f"train={train_loss:.4f} | val={val_loss:.4f} | "
                      f"lr={lr_now:.1e} | {elapsed:.0f}s")
            if nan_count > 0:
                status += f" | nan_skipped={nan_count}"
            if is_best:
                status += " *BEST*"
            print(status)

            log_entry = {
                'epoch': epoch + 1, 'train_loss': round(train_loss, 6),
                'val_loss': round(val_loss, 6), 'lr': round(lr_now, 8),
                'elapsed_sec': round(elapsed, 1), 'nan_batches': nan_count,
                'is_best': is_best}
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            if wandb_run is not None:
                import wandb
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss,
                           'lr': lr_now}, step=epoch + 1)

            should_save = (is_best or (epoch + 1) % checkpoint_every == 0
                           or epoch == epochs - 1)
            if should_save:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                                best_val_loss, train_loss, val_loss,
                                checkpoint_dir, is_best, distributed)

        if distributed:
            bvl = torch.tensor([best_val_loss], device=device)
            dist.broadcast(bvl, src=0)
            best_val_loss = bvl.item()

    if hasattr(train_ds, 'close_handles'):
        train_ds.close_handles()
    if hasattr(val_ds, 'close_handles'):
        val_ds.close_handles()
    if wandb_run is not None:
        import wandb
        wandb.finish()
    if is_main_process():
        print(f"\nDone. Best val_loss: {best_val_loss:.4f}")
        print(f"Checkpoints: {checkpoint_dir}")


# ---- Hydra integration ----

def load_config(config_dir: str, config_name: str = "config",
                overrides: list = None):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg


def train_from_config(cfg, rank=None, world_size_arg=None, port=None):
    """Map Hydra config to train() arguments.
    For deep ensemble configs, trains one model per seed."""
    is_ensemble = cfg.detector.get('ensemble_size', 1) > 1
    seeds = list(cfg.detector.seeds) if is_ensemble else [cfg.seed]
    base_ckpt_dir = cfg.checkpoint_dir

    for i, seed in enumerate(seeds):
        ckpt_dir = (f"{base_ckpt_dir}/member_{seed}"
                    if is_ensemble else base_ckpt_dir)
        if is_ensemble and (rank is None or rank == 0):
            print(f"\n{'='*60}")
            print(f"Ensemble member {i+1}/{len(seeds)}, seed={seed}")
            print(f"{'='*60}\n")

        train(
            data_dir=cfg.data.data_dir, val_dir=cfg.data.val_dir,
            checkpoint_dir=ckpt_dir, resume_from=cfg.resume_from,
            epochs=cfg.training.epochs, batch_size=cfg.training.batch_size,
            lr=cfg.training.lr, num_workers=cfg.training.num_workers,
            acceleration=cfg.data.acceleration,
            center_fraction=cfg.data.center_fraction,
            mask_type=cfg.data.mask_type,
            max_train_volumes=cfg.data.max_train_volumes,
            max_val_volumes=cfg.data.max_val_volumes,
            checkpoint_every=cfg.training.checkpoint_every,
            seed=seed, deterministic=cfg.deterministic,
            skip_nan_batches=cfg.training.skip_nan_batches,
            synthetic=cfg.synthetic,
            model_channels=list(cfg.detector.channels),
            model_dropout_p=cfg.detector.dropout_p,
            do_profile=cfg.get('do_profile', False),
            profile_dir=cfg.get('profile_dir', './profile_traces'),
            max_grad_norm=cfg.training.get('max_grad_norm', None),
            warmup_epochs=cfg.training.get('warmup_epochs', 0),
            use_wandb=cfg.get('use_wandb', False),
            wandb_project=cfg.get('wandb_project', 'mri-hallucination-physics'),
            rank=rank, world_size_arg=world_size_arg, port=port)


if __name__ == "__main__":
    config_dir = str(Path(__file__).resolve().parent / "configs")
    if not Path(config_dir).is_dir():
        config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    if not Path(config_dir).is_dir():
        raise FileNotFoundError(
            f"Cannot find configs/ directory. Searched: "
            f"{Path(__file__).resolve().parent / 'configs'}, "
            f"{Path(__file__).resolve().parent.parent / 'configs'}")
    overrides = [a for a in sys.argv[1:] if not a.startswith("--local")]
    cfg = load_config(config_dir, overrides=overrides)
    train_from_config(cfg)
