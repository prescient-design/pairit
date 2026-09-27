import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import torch
import torch.utils.data
from torch.utils.data import random_split, Dataset
from beignet.datasets import ChEMBLDataset
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Sampler

from pairit.transforms import Transform

T = TypeVar("T")


class ChEMBLPairDataset(Dataset):
    """
    A dataset wrapper that creates pairs from ChEMBL molecules.
    
    This dataset takes the original ChEMBL dataset and modifies each sample
    to create a pair format: <cls>smiles1<sep>smiles2<eos>
    """
    
    def __init__(self, chembl_dataset: ChEMBLDataset):
        """
        Initialize the ChEMBL pair dataset.
        
        Parameters
        ----------
        chembl_dataset : ChEMBLDataset
            The original ChEMBL dataset to wrap
        """
        self.chembl_dataset = chembl_dataset
        self.transform = chembl_dataset.transform
        self.tokenizer = self.transform.tokenizer
        
        # Get special token IDs
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
    def __len__(self) -> int:
        return len(self.chembl_dataset)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a training example with pair format.
        
        Parameters
        ----------
        idx : int
            Index of the sample
            
        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing 'input_ids' and 'attention_mask' for the pair format
        """
        # Get the original sample
        original_sample = self.chembl_dataset[idx]
        
        # Decode the original input_ids to get the SMILES string
        # Remove <cls> and <eos> tokens to get just the SMILES
        original_input_ids = original_sample['input_ids'].squeeze(0)
        original_attention_mask = original_sample['attention_mask'].squeeze(0)
        
        # Find where the actual SMILES starts and ends (skip <cls> and <eos>)
        # <cls> is typically at position 0, <eos> is at the end
        smiles_start = 1  # Skip <cls>
        smiles_end = (original_attention_mask == 1).sum() - 1  # Stop before <eos>
        
        # Extract the SMILES tokens
        smiles_tokens = original_input_ids[smiles_start:smiles_end]
        
        # Create the pair format: <cls>smiles<sep>smiles<eos>
        # For now, we'll use the same SMILES for both parts (you can modify this later)
        pair_input_ids = torch.cat([
            torch.tensor([self.cls_token_id]),  # <cls>
            smiles_tokens,                       # first SMILES
            torch.tensor([self.sep_token_id]),  # <sep>
            smiles_tokens,                       # second SMILES (same as first for now)
            torch.tensor([self.eos_token_id])   # <eos>
        ])
        
        # Create attention mask for the pair
        pair_attention_mask = torch.ones_like(pair_input_ids)
        
        # Pad to the same length as the original if needed
        max_length = original_input_ids.shape[0]
        if len(pair_input_ids) < max_length:
            # Pad with pad tokens
            padding_length = max_length - len(pair_input_ids)
            pair_input_ids = torch.cat([
                pair_input_ids,
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])
            pair_attention_mask = torch.cat([
                pair_attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])
        elif len(pair_input_ids) > max_length:
            # Truncate if too long
            pair_input_ids = pair_input_ids[:max_length]
            pair_attention_mask = pair_attention_mask[:max_length]
            # Ensure <eos> is at the end
            pair_input_ids[-1] = self.eos_token_id
            pair_attention_mask[-1] = 1
        
        return {
            'input_ids': pair_input_ids.unsqueeze(0),  # Add batch dimension
            'attention_mask': pair_attention_mask.unsqueeze(0)  # Add batch dimension
        }


class ChEMBLPairLightningDataModule(LightningDataModule):
    def __init__(
        self,
        root: str | Path = None,
        *,
        download: bool = False,
        transform_fn: Callable | Transform | None = None,
        lengths: Sequence[float] | None = (0.9, 0.05, 0.05),
        generator: Generator | None = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Iterable | Sampler | None = None,
        batch_sampler: Iterable[Sequence] | Sampler[Sequence] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[T]], Any] | None = None,
        max_length: int = 512,
        pin_memory: bool = True,
        drop_last: bool = False,
        max_train_samples: Optional[Union[int, float]] = None,
    ) -> None:
        """
        Initialize the ChEMBLPairLightningDataModule.

        Parameters
        ----------
        root : str or Path, optional
            Root directory where the dataset subdirectory exists or, if `download` is True,
            the directory where the dataset subdirectory will be created and the dataset downloaded.
        download : bool, optional
            If True, download the dataset to the `root` directory (default: False).
            If the dataset is already downloaded, it is not redownloaded.
        transform_fn : Callable or Transform or None, optional
            Function or transform to apply to the dataset for tokenization (default: None).
        lengths : Sequence[float] or None, optional
            Fractions of splits to generate (default: (0.9, 0.05, 0.05)).
        generator : Generator or None, optional
            Generator used for the random permutation (default: None).
        seed : int, optional
            Desired seed. Value must be within the inclusive range [-0x8000000000000000, 0xFFFFFFFFFFFFFFFF]
            (default: 0xDEADBEEF). Otherwise, a RuntimeError is raised. Negative inputs are remapped to positive
            values with the formula 0xFFFFFFFFFFFFFFFF + seed.
        batch_size : int, optional
            Samples per batch (default: 1).
        shuffle : bool, optional
            If True, reshuffle datasets at every epoch (default: True).
        sampler : Iterable or Sampler or None, optional
            Strategy to draw samples from the dataset (default: None). Can be any Iterable with __len__ implemented.
            If specified, `shuffle` must be False.
        batch_sampler : Iterable[Sequence] or Sampler[Sequence] or None, optional
            `sampler`, but returns a batch of indices (default: None). Mutually exclusive with `batch_size`,
            `shuffle`, `sampler`, and `drop_last`.
        num_workers : int, optional
            Subprocesses to use (default: 0). 0 means that the datasets will be loaded in the main process.
        collate_fn : Callable[[list[T]], Any] or None, optional
            Merges samples to form a mini-batch of Tensor(s) (default: None).
        max_length : int, optional
            Maximum length of the sequences (default: 512).
        pin_memory : bool, optional
            If True, Tensors are copied to the device's (e.g., CUDA) pinned memory before returning them (default: True).
        drop_last : bool, optional
            If True, drop the last incomplete batch, if the dataset size is not divisible by the batch size (default: False).
            If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
        max_train_samples: int or float, optional
            Maximum number of training samples to use (for debugging).
            Can be an integer > 1 (absolute count) or a float between 0 and 1 (fraction of dataset).
            If float, it will be converted to integer by rounding (fraction * dataset_length).
        """
        super().__init__()

        if generator is None:
            generator = Generator().manual_seed(seed)

        self._root = root
        self._download = download
        self._lengths = lengths
        self._generator = generator
        self._seed = seed
        self._batch_size = batch_size
        self._max_length = max_length
        self._shuffle = shuffle
        self._sampler = sampler
        self._batch_sampler = batch_sampler
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._dataset = None
        self._transform_fn = transform_fn
        self._max_train_samples = max_train_samples

    def prepare_data(self) -> None:
        # Create the original ChEMBL dataset
        original_dataset = ChEMBLDataset(
            root=self._root,
            download=self._download,
            transform=self._transform_fn,
        )
        # Wrap it with our pair dataset
        self._dataset = ChEMBLPairDataset(original_dataset)

    def setup(self, stage: str = "fit") -> None:
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        if self._dataset is None:
            self.prepare_data()

        if stage == "fit" or stage == "test":
            (
                self._train_dataset,
                self._val_dataset,
                self._test_dataset,
            ) = torch.utils.data.random_split(
                self._dataset,
                lengths=self._lengths,
                generator=self._generator,
            )

        # hl adding
        if stage == "predict":
            self._predict_dataset = self._dataset
        
        if self._max_train_samples is not None:
            # Convert fractional value (0-1) to integer count if needed
            if isinstance(self._max_train_samples, float):
                if not (0 < self._max_train_samples <= 1):
                    raise ValueError(f"max_train_samples as float must be between 0 and 1, got {self._max_train_samples}")
                max_samples = round(self._max_train_samples * len(self._train_dataset))
            else:
                if self._max_train_samples <= 0:
                    raise ValueError(f"max_train_samples as integer must be > 0, got {self._max_train_samples}")
                max_samples = self._max_train_samples
            
            self._train_dataset, _ = random_split(self._train_dataset, [max_samples, len(self._train_dataset) - max_samples])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self._predict_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )
