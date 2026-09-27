import importlib
import importlib.metadata
from importlib.metadata import PackageNotFoundError

try:
    __version__ = importlib.metadata.version("pairit")
except PackageNotFoundError:
    __version__ = None

# Define modules that are part of the public API
__all__ = ["callbacks", "cmdline", "data", "evaluation", "hydra_config", "model"]

def __getattr__(name: str):
    """
    Lazy submodule imports.

    This keeps `import pairit` lightweight (so utility scripts can import tokenizers/transforms
    without requiring optional training dependencies like `lightning`).
    """
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
