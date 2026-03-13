import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def extract_image_outputs(outputs: List[Dict[str, Any]]) -> List[bytes]:
    """Return raw PNG bytes for all image outputs in a cell."""
    images: List[bytes] = []
    for out in outputs:
        data = out.get("data") or {}
        img_b64 = data.get("image/png")
        if not img_b64:
            continue
        # image/png can be a single string or list of strings
        if isinstance(img_b64, list):
            img_b64 = "".join(img_b64)
        try:
            images.append(base64.b64decode(img_b64))
        except Exception:
            continue
    return images


def externalize_figures(
    src_path: Path, dst_path: Path, figs_dir: Path
) -> Tuple[int, int]:
    """Create a copy of the notebook with no embedded images.

    All image/png outputs are saved as separate PNG files in `figs_dir`,
    and markdown cells referencing those files are inserted after the
    corresponding code cells.

    Returns (num_cells_with_images, num_images_saved).
    """
    nb = json.loads(src_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    figs_dir.mkdir(parents=True, exist_ok=True)

    new_cells: List[Dict[str, Any]] = []
    fig_index = 1
    cells_with_images = 0

    for cell in cells:
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue

        outputs = cell.get("outputs") or []
        images = extract_image_outputs(outputs)

        # Always drop outputs from the code cell in the new notebook
        cell = dict(cell)  # shallow copy
        cell["outputs"] = []
        cell["execution_count"] = None
        new_cells.append(cell)

        if not images:
            continue

        cells_with_images += 1
        for img_bytes in images:
            fig_name = f"fig_{fig_index:03d}.png"
            fig_path = figs_dir / fig_name
            fig_path.write_bytes(img_bytes)

            # Insert a markdown cell that references the external image
            rel_path = figs_dir.name + "/" + fig_name
            md_cell: Dict[str, Any] = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"![Figure {fig_index}]({rel_path})\n"],
            }
            new_cells.append(md_cell)

            fig_index += 1

    nb["cells"] = new_cells
    # Remove widget metadata if present
    nb.get("metadata", {}).pop("widgets", None)

    dst_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return cells_with_images, fig_index - 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a GitHub-friendly notebook by moving embedded PNG figures "
            "to separate image files and linking them via markdown."
        )
    )
    parser.add_argument("path", help="Path to the source .ipynb notebook")
    parser.add_argument(
        "--out",
        help="Output notebook path (default: <stem>_linked.ipynb next to source)",
    )
    parser.add_argument(
        "--figs-dir",
        help=(
            "Directory (relative to notebook) where extracted PNG figures are saved "
            "(default: <stem>_figs)"
        ),
    )
    args = parser.parse_args()

    src = Path(args.path)
    if not src.exists():
        raise SystemExit(f"Source notebook not found: {src}")

    if args.out:
        dst = Path(args.out)
    else:
        dst = src.with_name(src.stem + "_linked.ipynb")

    if args.figs_dir:
        figs_dir = Path(args.figs_dir)
        if not figs_dir.is_absolute():
            figs_dir = dst.parent / figs_dir
    else:
        figs_dir = dst.parent / f"{src.stem}_figs"

    cells_with_images, num_images = externalize_figures(src, dst, figs_dir)
    print(
        f"Written linked notebook to: {dst}\n"
        f"Extracted {num_images} images from {cells_with_images} code cells into: {figs_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

