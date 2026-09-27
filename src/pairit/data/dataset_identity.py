"""
Dataset identity prompting utilities.

We prefix molecule-pair examples with two special tokens:
  1) A dataset-start token: "<ds>"
  2) A dataset-id token:    "<ds_{id}>"

Important compatibility note:
- These tokens are intended to be appended to the tokenizer vocabulary via
  `add_special_tokens`, so they will NOT overlap / reassign existing SMILES token IDs.
"""

from __future__ import annotations

from dataclasses import dataclass


DATASET_START_TOKEN = "<ds>"


def dataset_id_to_token(dataset_id: int) -> str:
    if not isinstance(dataset_id, int):
        raise TypeError(f"dataset_id must be int, got {type(dataset_id)}")
    if dataset_id < 0:
        raise ValueError(f"dataset_id must be >= 0, got {dataset_id}")
    return f"<ds_{dataset_id}>"


def dataset_prompt_text(dataset_id: int) -> str:
    """
    Returns the text prefix that should appear AFTER <cls>.

    IMPORTANT: do NOT include whitespace.
    This SMILES tokenizer treats spaces/newlines as unknown tokens (<unk>), which would
    introduce extra tokens between the dataset identity tokens and the SMILES tokens.

    Adjacent special tokens are handled correctly, so we use a compact prefix:
      "<ds><ds_{id}>"
    """
    return f"{DATASET_START_TOKEN}{dataset_id_to_token(dataset_id)}"


@dataclass(frozen=True)
class DatasetIdentitySpec:
    name: str
    dataset_id: int


