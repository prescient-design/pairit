"""Location of the bundled assets/ tree.

The data files, oracle joblibs, baseline generations, and model checkpoints that
this package references at runtime live under ``assets/`` at the repository
root. Nothing there is committed to git; each entry has to be materialised on
the local disk before the code that consumes it will succeed.

Import ``ASSETS`` and build paths relative to it rather than hardcoding
per-user filesystem locations::

    from pairit._assets import ASSETS
    evaluator_dir = ASSETS / "oracles" / "moleculenet" / "evaluators"

Hydra configs can reach the same tree via the ``${pairit_assets:...}`` resolver
registered on import::

    path: ${pairit_assets:data/moleculenet/bace/train/molecules.parquet}

By default ``ASSETS`` resolves to ``assets/`` at the repository root
checkout. To store the ~8 GB bundle outside the repo (network filesystem,
scratch disk, etc.) set the ``PAIRIT_ASSETS_DIR`` environment variable::

    export PAIRIT_ASSETS_DIR=/data/pairit_release/assets
"""

import os
from pathlib import Path

# Default: __file__ resolves to .../src/pairit/_assets.py, so
# parents[2] is .../ the repository root, which contains assets/. Override with the
# PAIRIT_ASSETS_DIR env var to point the bundle at any directory on disk.
_DEFAULT_ASSETS = Path(__file__).resolve().parents[2] / "assets"
ASSETS = Path(os.environ["PAIRIT_ASSETS_DIR"]) if "PAIRIT_ASSETS_DIR" in os.environ else _DEFAULT_ASSETS


def _register_pairit_assets_resolver() -> None:
    """Register a Hydra/OmegaConf resolver that expands ``${pairit_assets:<sub>}``.

    Idempotent: safe to call more than once. No-op when OmegaConf is not
    installed (importing pairit for lightweight utilities should not require it).
    """
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return
    if not OmegaConf.has_resolver("pairit_assets"):
        OmegaConf.register_new_resolver(
            "pairit_assets", lambda subpath: str(ASSETS / subpath)
        )


_register_pairit_assets_resolver()
