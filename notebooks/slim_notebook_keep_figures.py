import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def keep_only_figure_outputs(
    outputs: List[Dict[str, Any]], max_remaining: int
) -> List[Dict[str, Any]]:
    """Return only outputs that contain image data (e.g. PNG) so figures still show.

    `max_remaining` is a global budget for how many image outputs we still allow
    across the whole notebook. Once it reaches zero, this returns an empty list.
    """
    if max_remaining <= 0:
        return []

    kept: List[Dict[str, Any]] = []

    for out in outputs:
        if max_remaining <= 0:
            break

        output_type = out.get("output_type")
        if output_type not in ("display_data", "execute_result"):
            # Drop streams, errors, etc. to keep size small
            continue

        data = out.get("data") or {}
        has_image = any(key.startswith("image/") for key in data.keys())
        if has_image:
            kept.append(out)
            max_remaining -= 1

    return kept


def slim_notebook(src_path: Path, dst_path: Path, max_images: int) -> None:
    nb = json.loads(src_path.read_text(encoding="utf-8"))

    # Remove widget metadata if present
    nb.get("metadata", {}).pop("widgets", None)

    # Global budget for how many image outputs we keep total
    remaining = max_images

    # For each code cell, keep only figure outputs, honoring global budget
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs") or []
        kept_outputs = keep_only_figure_outputs(outputs, remaining)
        remaining -= len(kept_outputs)
        cell["outputs"] = kept_outputs
        # Optional: keep execution_count so GitHub shows In [ ] labels
        # If you prefer a "cleaner" look, set it to None instead.

    dst_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a GitHub-friendly notebook that keeps only figure outputs "
            "(image displays) and drops heavy logs and non-image outputs."
        )
    )
    parser.add_argument(
        "path",
        help="Path to the source .ipynb notebook",
    )
    parser.add_argument(
        "--out",
        help="Output notebook path (default: <stem>_figs.ipynb next to source)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help=(
            "Maximum number of figure outputs (images) to keep across the whole "
            "notebook (default: 20)."
        ),
    )
    args = parser.parse_args()

    src = Path(args.path)
    if not src.exists():
        raise SystemExit(f"Source notebook not found: {src}")

    if args.out:
        dst = Path(args.out)
    else:
        dst = src.with_name(src.stem + "_figs.ipynb")

    slim_notebook(src, dst, max_images=args.max_images)
    print(f"Written slim notebook with figures to: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

