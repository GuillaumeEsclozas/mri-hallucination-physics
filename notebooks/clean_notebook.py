import argparse
import json
from pathlib import Path


def clean_notebook(src_path: Path, dst_path: Path) -> None:
    nb = json.loads(src_path.read_text(encoding="utf-8"))

    # Remove widget metadata if present
    nb.get("metadata", {}).pop("widgets", None)

    # Strip outputs and execution counts from all code cells
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    dst_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a GitHub-friendly version of a Jupyter notebook by "
            "removing outputs, execution counts, and widget metadata."
        )
    )
    parser.add_argument(
        "path",
        help="Path to the source .ipynb notebook",
    )
    parser.add_argument(
        "--out",
        help="Output notebook path (default: <stem>_clean.ipynb next to source)",
    )
    args = parser.parse_args()

    src = Path(args.path)
    if not src.exists():
        raise SystemExit(f"Source notebook not found: {src}")

    if args.out:
        dst = Path(args.out)
    else:
        dst = src.with_name(src.stem + "_clean.ipynb")

    clean_notebook(src, dst)
    print(f"Written cleaned notebook to: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

