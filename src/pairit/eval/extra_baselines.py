"""Hook for baseline families registered outside ``comparison_matrix``.

``comparison_matrix`` and ``scripts/print_latex_tables`` merge every
``ExtraBaseline`` listed in the optional module
``pairit.eval._internal_baselines`` (attribute ``EXTRA_BASELINES``) into their
spec list, loader dispatch, ordering, colours and labels. Without that module
the comparison runs the built-in baselines only.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    import pandas as pd

    from .comparison_matrix import SourceSpec

_OPTIONAL_MODULE = "pairit.eval._internal_baselines"


@dataclass(frozen=True)
class ExtraBaseline:
    family: str
    # Returns the family's SourceSpecs (missing files already skipped).
    build_specs: Callable[[], List["SourceSpec"]]
    # source_kind -> loader returning a CANONICAL_COLUMNS frame.
    loaders: Dict[str, Callable[["SourceSpec"], "pd.DataFrame"]]
    # Sort key in the method table, same scale as comparison_matrix's family order.
    family_order: int
    # Plot and LaTeX rows are inserted right after this family's rows.
    after_family: str
    color: str
    plot_labels: Dict[str, str] = field(default_factory=dict)
    scatter_labels: Dict[str, str] = field(default_factory=dict)
    latex_names: Dict[str, str] = field(default_factory=dict)


@lru_cache(maxsize=None)
def extra_baselines() -> Tuple[ExtraBaseline, ...]:
    try:
        module = importlib.import_module(_OPTIONAL_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != _OPTIONAL_MODULE:
            raise  # the module exists but one of its own imports is missing
        return ()
    return tuple(module.EXTRA_BASELINES)


def insert_after(items: List[str], anchor: str, new: str) -> List[str]:
    """Return ``items`` with ``new`` placed right after ``anchor`` (appended if absent)."""
    if new in items:
        return list(items)
    out = list(items)
    out.insert(out.index(anchor) + 1 if anchor in out else len(out), new)
    return out
