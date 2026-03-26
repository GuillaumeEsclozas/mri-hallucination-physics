"""
Download and validate fastMRI single coil knee data.
Sets up the directory structure expected by the training scripts.
"""

import os
import sys
import hashlib
from pathlib import Path


EXPECTED_STRUCTURE = {
    'singlecoil_train': {'min_files': 900, 'extension': '.h5'},
    'singlecoil_val': {'min_files': 190, 'extension': '.h5'},
}


def validate_directory(data_root: str):
    root = Path(data_root)
    if not root.exists():
        print(f"ERROR: {root} does not exist.")
        print("Download the fastMRI dataset from https://fastmri.med.nyu.edu/")
        print(f"Extract to: {root}")
        return False

    all_ok = True
    for subdir, spec in EXPECTED_STRUCTURE.items():
        path = root / subdir
        if not path.exists():
            print(f"MISSING: {path}")
            all_ok = False
            continue

        h5_files = list(path.glob(f"*{spec['extension']}"))
        n = len(h5_files)
        expected = spec['min_files']
        status = "OK" if n >= expected else "WARNING"
        print(f"  {subdir}: {n} files ({status}, expected >= {expected})")
        if n < expected:
            all_ok = False

    return all_ok


def compute_checksum(filepath: str, algorithm: str = 'md5',
                     chunk_size: int = 8192) -> str:
    h = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def setup_scratch(data_root: str, scratch_dir: str):
    """Symlink or copy data to scratch for fast I/O on HPC."""
    src = Path(data_root)
    dst = Path(scratch_dir)
    if dst.exists():
        print(f"Scratch directory already exists: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        os.symlink(str(src.resolve()), str(dst))
        print(f"Symlinked {src} -> {dst}")
    else:
        print(f"Source {src} not found. Download data first.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_data.py <data_root> [scratch_dir]")
        print("  data_root:   path containing singlecoil_train/ and singlecoil_val/")
        print("  scratch_dir: optional fast storage path for HPC (creates symlink)")
        sys.exit(1)

    data_root = sys.argv[1]
    print(f"Validating data at: {data_root}\n")
    ok = validate_directory(data_root)

    if len(sys.argv) >= 3:
        scratch = sys.argv[2]
        print(f"\nSetting up scratch at: {scratch}")
        setup_scratch(data_root, scratch)

    if ok:
        print("\nData validation passed.")
    else:
        print("\nData validation found issues. See above.")
        sys.exit(1)
