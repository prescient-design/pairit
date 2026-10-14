"""
Checkpoint compatibility for PyTorch 2.6+

PyTorch 2.6 changed the default of `weights_only` from False to True in torch.load().
This module registers all pairit classes that may be stored in checkpoints as safe globals.

Import this module early (before any checkpoint loading) to enable loading pairit checkpoints.

Historical prompting checkpoints pickle two classes under the former ``lobster``
package name. Only those class paths are aliased; this is not a Lobster dependency
or a general redirect of its modules.
"""
import sys
from types import ModuleType

import torch

# List of all pairit classes that may be stored in checkpoints
_SAFE_GLOBALS = []


def _register_legacy_class_alias(module_name: str, cls: type) -> None:
    """Let ordinary pickle resolve one renamed class without importing Lobster."""
    parts = module_name.split(".")
    for index in range(1, len(parts) + 1):
        name = ".".join(parts[:index])
        if name not in sys.modules:
            module = ModuleType(name)
            if index < len(parts):
                module.__path__ = []
            sys.modules[name] = module
            if index > 1:
                setattr(sys.modules[".".join(parts[:index - 1])], parts[index - 1], module)
    module = sys.modules[module_name]
    if not hasattr(module, cls.__name__):
        setattr(module, cls.__name__, cls)
    # PyTorch's weights-only unpickler resolves names from this explicit map.
    _SAFE_GLOBALS.append((cls, f"{module_name}.{cls.__name__}"))


try:
    from pairit.transforms._tokenizer_transform import TokenizerTransform
    _SAFE_GLOBALS.append(TokenizerTransform)
    _register_legacy_class_alias("lobster.transforms._tokenizer_transform", TokenizerTransform)
except ImportError:
    pass

try:
    from pairit.tokenization._smiles_tokenizer2 import SmilesTokenizer2Fast
    _SAFE_GLOBALS.append(SmilesTokenizer2Fast)
    _register_legacy_class_alias("lobster.tokenization._smiles_tokenizer2", SmilesTokenizer2Fast)
except ImportError:
    print("SmilesTokenizer2Fast not found")
    pass

# Add tokenizers library classes that may be in checkpoints
try:
    from tokenizers import AddedToken, Tokenizer
    _SAFE_GLOBALS.extend([Tokenizer, AddedToken])
    # Also add common tokenizers subclasses
    from tokenizers.models import Model
    from tokenizers.pre_tokenizers import PreTokenizer
    from tokenizers.processors import PostProcessor
    from tokenizers.normalizers import Normalizer
    _SAFE_GLOBALS.extend([Model, PreTokenizer, PostProcessor, Normalizer])
except ImportError:
    pass

# Add transformers tokenizer classes that may be in checkpoints
try:
    from transformers import PreTrainedTokenizerFast
    _SAFE_GLOBALS.append(PreTrainedTokenizerFast)
except ImportError:
    pass

# Add OmegaConf classes that may be in checkpoints (stored in hyper_parameters).
# Lightning stores the full hyper_parameters dict which contains OmegaConf internals.
# All of these must be allowlisted for weights_only=True loading (required for
# preemption resume via trainer.fit(ckpt_path=...)).
try:
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode, ValueNode
    _SAFE_GLOBALS.extend([DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode, ValueNode])
except ImportError:
    pass

# typing.Any appears in OmegaConf node metadata stored in checkpoints
import typing
_SAFE_GLOBALS.append(typing.Any)

# Add any other classes that might be pickled in checkpoints here:
# try:
#     from pairit.some_module import SomeClass
#     _SAFE_GLOBALS.append(SomeClass)
# except ImportError:
#     pass

if _SAFE_GLOBALS:
    torch.serialization.add_safe_globals(_SAFE_GLOBALS)



