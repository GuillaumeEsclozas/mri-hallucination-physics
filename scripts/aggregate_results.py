"""
Collect results from multiple training runs into a summary table.
Scans checkpoint directories for checkpoint_best.pt and extracts metrics.
Handles both single run (checkpoint in root) and sweep (subdirectories).
"""

import csv
import sys
from pathlib import Path

import torch


def _extract_row(ckpt_path: Path, run_name: str) -> dict:
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  Error loading {ckpt_path}: {e}")
        return None

    return {
        'run': run_name,
        'epoch': ckpt.get('epoch', -1) + 1,
        'train_loss': round(ckpt.get('train_loss', float('nan')), 5),
        'val_loss': round(ckpt.get('val_loss', float('nan')), 5),
        'best_val_loss': round(ckpt.get('best_val_loss', float('nan')), 5),
    }


def aggregate(results_root: str, output_csv: str):
    root = Path(results_root)
    rows = []

    # Check if root itself contains a checkpoint (single run case)
    root_ckpt = root / 'checkpoint_best.pt'
    if root_ckpt.exists():
        row = _extract_row(root_ckpt, root.name)
        if row:
            rows.append(row)

    # Scan subdirectories (sweep case)
    for ckpt_dir in sorted(root.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        best_ckpt = ckpt_dir / 'checkpoint_best.pt'
        if not best_ckpt.exists():
            continue
        row = _extract_row(best_ckpt, ckpt_dir.name)
        if row:
            rows.append(row)

    if not rows:
        print("No results found.")
        return

    rows.sort(key=lambda r: r['best_val_loss'])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'Run':<30} {'Epoch':>6} {'Train':>10} {'Val':>10} {'Best':>10}")
    print("=" * 70)
    for r in rows:
        print(f"{r['run']:<30} {r['epoch']:>6} "
              f"{r['train_loss']:>10.5f} {r['val_loss']:>10.5f} "
              f"{r['best_val_loss']:>10.5f}")
    print(f"\nSaved to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python aggregate_results.py <results_root> <output.csv>")
        sys.exit(1)
    aggregate(sys.argv[1], sys.argv[2])
