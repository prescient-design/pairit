"""Download the public source datasets used by PairIT.

Two upstream projects publish the exact data and splits used in the paper:

* MoleculeNet BACE / BBBP / HIV, with their scaffold splits, come from
  Graph-DiT (https://github.com/liugangcode/Graph-DiT).
* ZINC-250K, with the validation index used for pretraining, comes from
  MOOD (https://github.com/SeulLee05/MOOD).

Nothing here is redistributed; the files are fetched from the original
repositories at run time.

Usage::

    python scripts/download_data.py --dest data/raw
    python scripts/download_data.py --dest data/raw --dataset bace
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

GRAPH_DIT = "https://raw.githubusercontent.com/liugangcode/Graph-DiT/main"
MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main"

# dataset -> (url, destination path relative to --dest)
SOURCES: dict[str, list[tuple[str, str]]] = {
    "bace": [
        (f"{GRAPH_DIT}/data/raw/bace_b.csv.gz", "moleculenet/bace_b.csv.gz"),
        (f"{GRAPH_DIT}/data/split/bace_b_split.pt", "moleculenet/bace_b_split.pt"),
    ],
    "bbbp": [
        (f"{GRAPH_DIT}/data/raw/bbbp_b.csv.gz", "moleculenet/bbbp_b.csv.gz"),
        (f"{GRAPH_DIT}/data/split/bbbp_b_split.pt", "moleculenet/bbbp_b_split.pt"),
    ],
    "hiv": [
        (f"{GRAPH_DIT}/data/raw/hiv_b.csv.gz", "moleculenet/hiv_b.csv.gz"),
        (f"{GRAPH_DIT}/data/split/hiv_b_split.pt", "moleculenet/hiv_b_split.pt"),
    ],
    "zinc": [
        (f"{MOOD}/data/zinc250k.csv", "zinc/zinc250k.csv"),
        (f"{MOOD}/data/valid_idx_zinc250k.json", "zinc/valid_idx_zinc250k.json"),
    ],
}


def download(url: str, target: Path, overwrite: bool = False) -> None:
    if target.exists() and not overwrite:
        print(f"  skip (exists)  {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            tmp.write_bytes(response.read())
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"failed to download {url}: {exc}") from exc
    tmp.replace(target)
    digest = hashlib.md5(target.read_bytes()).hexdigest()
    print(f"  {target}  ({target.stat().st_size:,} bytes, md5 {digest})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw"),
        help="directory to download into (default: data/raw)",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(SOURCES),
        action="append",
        help="limit to one dataset; repeatable (default: all)",
    )
    parser.add_argument("--overwrite", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    datasets = args.dataset or sorted(SOURCES)
    for name in datasets:
        print(f"{name}:")
        for url, relative in SOURCES[name]:
            download(url, args.dest / relative, overwrite=args.overwrite)

    print(f"\nDone. Next: python scripts/prepare_molecules.py --raw {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
