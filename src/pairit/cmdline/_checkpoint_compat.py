"""
Checkpoint compatibility for PyTorch 2.6+

PyTorch 2.6 changed the default of `weights_only` from False to True in torch.load().
This module registers all pairit classes that may be stored in checkpoints as safe globals.

Import this module early (before any checkpoint loading) to enable loading pairit checkpoints.
"""
import torch

# List of all pairit classes that may be stored in checkpoints
_SAFE_GLOBALS = []

try:
    from pairit.transforms._tokenizer_transform import TokenizerTransform
    _SAFE_GLOBALS.append(TokenizerTransform)
except ImportError:
    pass

try:
    from pairit.tokenization._smiles_tokenizer2 import SmilesTokenizer2Fast
    _SAFE_GLOBALS.append(SmilesTokenizer2Fast)
except ImportError:
    print("SmilesTokenizer2Fast not found")
    pass

# Add tokenizers library classes that may be in checkpoints
try:
    from tokenizers import Tokenizer
    _SAFE_GLOBALS.append(Tokenizer)
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



