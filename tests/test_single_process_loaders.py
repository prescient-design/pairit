"""Regression for sampling/training with data.num_workers=0."""

import pytest
import torch
from torch.utils.data import TensorDataset

from pairit.data._molecule_improvement_datamodule import MoleculeImprovementLightningDataModule


@pytest.mark.parametrize("split", ["train", "val", "test", "predict"])
def test_single_process_loader_reads_all_batches(tmp_path, split):
    dm = MoleculeImprovementLightningDataModule(
        root=str(tmp_path), num_workers=0, batch_size=2,
        pin_memory=False, shuffle=False, drop_last=False,
    )
    dataset = TensorDataset(torch.arange(5))
    dm._train_dataset = dm._val_dataset = dm._test_dataset = dataset
    loader = getattr(dm, f"{split}_dataloader")()
    assert torch.cat([batch[0] for batch in loader]).tolist() == list(range(5))
