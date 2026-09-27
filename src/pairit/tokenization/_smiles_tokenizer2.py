import importlib.resources
from pathlib import Path

from tokenizers import Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from ._load_vocab_file import load_vocab_file
from ._make_pretrained_tokenizer_fast import make_pretrained_tokenizer_fast
DATASET_START_TOKEN = "<ds>"


def dataset_id_to_token(dataset_id: int) -> str:
    if not isinstance(dataset_id, int):
        raise TypeError(f"dataset_id must be int, got {type(dataset_id)}")
    if dataset_id < 0:
        raise ValueError(f"dataset_id must be >= 0, got {dataset_id}")
    return f"<ds_{dataset_id}>"

SMILES_REGEX_PATTERN = (
    r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
)

PRETRAINED_TOKENIZER_PATH = importlib.resources.files("pairit") / "assets" / "smiles_tokenizer2_with_dataset_tokens"
VOCAB_PATH = PRETRAINED_TOKENIZER_PATH / "vocab.txt"


print(f'PRETRAINED_TOKENIZER_PATH: {PRETRAINED_TOKENIZER_PATH}')

def _make_smiles_tokenizer2(save_dirpath: str | Path | None = PRETRAINED_TOKENIZER_PATH) -> PreTrainedTokenizerFast:
    """Create PreTrainedTokenizerFast for SMILES Regex tokenization.

    Usage:
    ```python
    tokenizer = _make_ume_tokenizer(save_dirpath=PRETRAINED_TOKENIZER_PATH)

    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(PRETRAINED_TOKENIZER_PATH)
    """

    cls_token = "<cls>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = "<pad>"
    sep_token = "<sep>"
    mask_token = "<mask>"

    vocab = load_vocab_file(VOCAB_PATH)
    vocab = {v: k for k, v in enumerate(vocab)}

    tokenizer_model = WordLevel(vocab=vocab, unk_token=unk_token)

    pre_tokenizer = Split(pattern=Regex(SMILES_REGEX_PATTERN), behavior="isolated")

    post_processor = TemplateProcessing(
        single=f"{cls_token} $A {eos_token}",
        pair=f"{cls_token} $A {sep_token} $B:1 {eos_token}:1", # I think this is being used?
        special_tokens=[
            (cls_token, 0),
            (eos_token, 2),
            (sep_token, 5),
        ],
    )

    return make_pretrained_tokenizer_fast(
        tokenizer_model=tokenizer_model,
        save_dirpath=save_dirpath,
        pre_tokenizer=pre_tokenizer,
        post_processor=post_processor,
        cls_token=cls_token,
        eos_token=eos_token,
        unk_token=unk_token,
        sep_token=sep_token,
        mask_token=mask_token,
        pad_token=pad_token,
        bos_token=None,
    )


class SmilesTokenizer2Fast(PreTrainedTokenizerFast):
    padding_side = "right"
    truncation_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        *,
        inherit_special_tokens: list[str] | None = None,
        add_dataset_tokens: bool = False,
        max_dataset_id: int | None = None,
        add_property_prompt_tokens: bool = False,
        property_prompt_tokens: dict | None = None,
        add_distance_prompt_tokens: bool = False,
        distance_prompt_tokens: dict | None = None,
    ):
        super().__init__(
            tokenizer_file=str(PRETRAINED_TOKENIZER_PATH / "tokenizer.json"),
            bos_token=None,
            eos_token="<eos>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
        )

        # ---------------- inherited tokens from checkpoint (MUST come first) ----------------
        # These tokens are inherited from a checkpoint's tokenizer to ensure compatibility.
        # They are added BEFORE any other tokens to preserve the exact token IDs from the checkpoint.
        # We track them in a set so subsequent sections know these are valid special tokens.
        _inherited_token_set: set[str] = set()
        if inherit_special_tokens:
            _inherited_token_set = set(inherit_special_tokens)
            vocab = self.get_vocab()
            existing_additional = set(getattr(self, "additional_special_tokens", None) or [])
            existing_special = set(getattr(self, "all_special_tokens", None) or [])

            to_add: list[str] = []
            already_present: list[str] = []
            for tok in inherit_special_tokens:
                if tok in vocab:
                    already_present.append(tok)
                    continue
                if tok in existing_additional or tok in existing_special:
                    already_present.append(tok)
                    continue
                to_add.append(tok)

            if to_add:
                self.add_special_tokens({"additional_special_tokens": to_add})

            # Store for diagnostics (can be accessed later if needed)
            self._inherited_tokens_added = to_add
            self._inherited_tokens_already_present = already_present

        if add_dataset_tokens:
            if max_dataset_id is None:
                raise ValueError("max_dataset_id must be provided when add_dataset_tokens=True")
            if not isinstance(max_dataset_id, int) or max_dataset_id < 0:
                raise ValueError(f"max_dataset_id must be a non-negative int, got {max_dataset_id!r}")

            desired = [DATASET_START_TOKEN] + [dataset_id_to_token(i) for i in range(max_dataset_id + 1)]

            vocab = self.get_vocab()
            existing_additional = set(getattr(self, "additional_special_tokens", None) or [])
            existing_special = set(getattr(self, "all_special_tokens", None) or [])

            to_add: list[str] = []
            for tok in desired:
                if tok in vocab:
                    # Ensure we are not overlapping with a normal (non-special) token.
                    # If it's already present and marked special (or inherited), that's fine (id is stable).
                    if tok not in existing_additional and tok not in existing_special and tok not in _inherited_token_set:
                        raise ValueError(
                            f"Refusing to reuse existing non-special token {tok!r} for dataset prompting. "
                            "Choose different dataset token strings."
                        )
                    continue
                to_add.append(tok)

            if to_add:
                # This is append-only: new IDs are added after the existing vocab,
                # preserving all existing token IDs for checkpoint compatibility.
                self.add_special_tokens({"additional_special_tokens": to_add})

        # ---------------- property prompting tokens (append-only) ----------------
        if add_property_prompt_tokens:
            if property_prompt_tokens is None:
                raise ValueError("property_prompt_tokens must be provided when add_property_prompt_tokens=True")

            prop_names = property_prompt_tokens.get("property_names", None)
            if not prop_names:
                raise ValueError("property_prompt_tokens.property_names must be provided and non-empty")
            prop_names = [str(p) for p in prop_names]

            num_value_bins = int(property_prompt_tokens.get("num_value_bins", 0) or 0)
            num_delta_bins = int(property_prompt_tokens.get("num_delta_bins", 0) or 0)
            if num_value_bins <= 0:
                raise ValueError("property_prompt_tokens.num_value_bins must be a positive int")
            if num_delta_bins <= 0:
                raise ValueError("property_prompt_tokens.num_delta_bins must be a positive int")

            # Deterministic ordering is important for reproducibility.
            prop_names_sorted = sorted(set(prop_names))
            value_width = max(2, len(str(num_value_bins - 1)))
            delta_width = max(2, len(str(num_delta_bins - 1)))

            desired: list[str] = []
            desired.append("<pc>")
            desired.extend(["<pdir_up>", "<pdir_down>", "<pdir_any>"])
            for name in prop_names_sorted:
                desired.append(f"<p_{name}>")
                for b in range(num_value_bins):
                    desired.append(f"<p0_{name}_{b:0{value_width}d}>")
                for b in range(num_delta_bins):
                    desired.append(f"<pd_{name}_{b:0{delta_width}d}>")
            # Bounded delta tokens (atmost / atleast) — append-only for checkpoint compat
            for name in prop_names_sorted:
                for b in range(num_delta_bins):
                    desired.append(f"<pdmax_{name}_{b:0{delta_width}d}>")
                    desired.append(f"<pdmin_{name}_{b:0{delta_width}d}>")

            vocab = self.get_vocab()
            existing_additional = set(getattr(self, "additional_special_tokens", None) or [])
            existing_special = set(getattr(self, "all_special_tokens", None) or [])

            to_add: list[str] = []
            for tok in desired:
                if tok in vocab:
                    # Ensure we are not overlapping with a normal (non-special) token.
                    # If it's already present and marked special (or inherited), that's fine (id is stable).
                    if tok not in existing_additional and tok not in existing_special and tok not in _inherited_token_set:
                        raise ValueError(
                            f"Refusing to reuse existing non-special token {tok!r} for property prompting. "
                            "Choose different property prompt token strings."
                        )
                    continue
                to_add.append(tok)

            if to_add:
                # Append-only for checkpoint compatibility
                self.add_special_tokens({"additional_special_tokens": to_add})

        # ---------------- distance prompting tokens (append-only) ----------------
        if add_distance_prompt_tokens:
            if distance_prompt_tokens is None:
                raise ValueError("distance_prompt_tokens must be provided when add_distance_prompt_tokens=True")

            num_bins = int(distance_prompt_tokens.get("num_bins", 0) or 0)
            if num_bins <= 0:
                raise ValueError("distance_prompt_tokens.num_bins must be a positive int")

            bin_width = max(2, len(str(num_bins - 1)))

            desired: list[str] = []
            desired.append("<dc>")
            for b in range(num_bins):
                desired.append(f"<d_{b:0{bin_width}d}>")

            vocab = self.get_vocab()
            existing_additional = set(getattr(self, "additional_special_tokens", None) or [])
            existing_special = set(getattr(self, "all_special_tokens", None) or [])

            to_add: list[str] = []
            for tok in desired:
                if tok in vocab:
                    # Ensure we are not overlapping with a normal (non-special) token.
                    # If it's already present and marked special (or inherited), that's fine (id is stable).
                    if tok not in existing_additional and tok not in existing_special and tok not in _inherited_token_set:
                        raise ValueError(
                            f"Refusing to reuse existing non-special token {tok!r} for distance prompting. "
                            "Choose different distance prompt token strings."
                        )
                    continue
                to_add.append(tok)

            if to_add:
                # Append-only for checkpoint compatibility
                self.add_special_tokens({"additional_special_tokens": to_add})
