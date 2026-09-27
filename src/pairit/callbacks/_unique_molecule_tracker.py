"""Lightweight callback that tracks cumulative unique SMILES seen during training.

Logs ``unique_molecules_seen`` and ``total_molecules_seen`` to the active
logger (typically WandB) so overfitting can be correlated with dataset
saturation.
"""

import logging

import lightning.pytorch as pl

logger = logging.getLogger(__name__)


class UniqueMoleculeTracker(pl.Callback):
    """Track the number of unique canonical SMILES observed during training.

    For pair data each input contains two SMILES separated by the SEP token.
    Both molecules are canonicalized via RDKit before hashing.

    Parameters
    ----------
    log_every_n_steps : int
        How often (in training steps) to log metrics.  Defaults to 50.
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self._seen_hashes: set[int] = set()
        self._total_seen: int = 0
        self._log_every_n_steps = log_every_n_steps
        self._canonicalize = None  # lazy import
        self._tokenizer = None
        self._sep_token = None

    def _get_canonicalize(self):
        """Lazily import RDKit canonicalization to avoid import at module level."""
        if self._canonicalize is None:
            from rdkit import Chem

            def _canon(smiles: str) -> str | None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                return Chem.MolToSmiles(mol)

            self._canonicalize = _canon
        return self._canonicalize

    def _process_batch(self, batch):
        """Extract SMILES from a batch and update the seen set."""
        # Extract input_ids from batch — handle various batch formats
        if isinstance(batch, (list, tuple)):
            batch_dict = batch[0]
        else:
            batch_dict = batch
        if isinstance(batch_dict, dict):
            input_ids = batch_dict["input_ids"].squeeze()
        else:
            return

        if self._tokenizer is None:
            return

        tokenizer = self._tokenizer
        sep_token = self._sep_token

        canonicalize = self._get_canonicalize()
        for seq_ids in input_ids:
            decoded = tokenizer.decode(seq_ids.tolist(), skip_special_tokens=False)
            parts = decoded.split(sep_token)

            for part in parts:
                smiles = part.strip()
                for tok in [
                    "<cls>", "<eos>", "<pad>", "<mask>", "<unk>",
                    "[CLS]", "[EOS]", "[PAD]", "[MASK]", "[UNK]",
                ]:
                    smiles = smiles.replace(tok, "")
                smiles = smiles.replace(" ", "")

                if not smiles:
                    continue
                canon = canonicalize(smiles)
                if canon is not None:
                    self._seen_hashes.add(hash(canon))
                    self._total_seen += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Cache tokenizer on first call
        if self._tokenizer is None:
            self._tokenizer = pl_module.tokenizer
            self._sep_token = getattr(self._tokenizer, "sep_token", "<sep>") or "<sep>"

        # Process every batch for accurate tracking
        self._process_batch(batch)

        # Only log periodically to avoid log spam
        if batch_idx % self._log_every_n_steps == 0:
            pl_module.log("unique_molecules_seen", len(self._seen_hashes), prog_bar=True)
            pl_module.log("total_molecules_seen", self._total_seen)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["unique_molecule_tracker"] = {
            "seen_hashes": self._seen_hashes,
            "total_seen": self._total_seen,
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        state = checkpoint.get("unique_molecule_tracker")
        if state is not None:
            self._seen_hashes = state["seen_hashes"]
            self._total_seen = state["total_seen"]
            logger.info(
                "Restored UniqueMoleculeTracker: %d unique, %d total",
                len(self._seen_hashes),
                self._total_seen,
            )
