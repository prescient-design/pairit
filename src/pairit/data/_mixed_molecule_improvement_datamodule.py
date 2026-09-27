from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import lightning.pytorch as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data._utils.collate import default_collate


class _AddConstantFieldsDataset(Dataset):
    """
    Dataset wrapper that injects constant fields into each sample dict.

    Useful for mixed datasets so downstream callbacks can identify dataset source
    without relying on dataloader_idx.
    """

    def __init__(self, dataset: Dataset, *, fields: Mapping[str, Any]):
        self._dataset = dataset
        self._fields = dict(fields)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        item = self._dataset[idx]
        if isinstance(item, dict):
            out = dict(item)
            for k, v in self._fields.items():
                out.setdefault(k, v)
            return out
        return item


def _as_ordered_items(components: Any) -> List[Tuple[str, Any]]:
    """
    Normalize components input to a stable ordered list of (name, datamodule).
    """
    if isinstance(components, Mapping):
        return [(str(k), v) for k, v in components.items()]
    if isinstance(components, Sequence):
        out: List[Tuple[str, Any]] = []
        for i, dm in enumerate(components):
            out.append((f"component_{i}", dm))
        return out
    raise TypeError(f"components must be a mapping or sequence, got {type(components)}")


def _subset_dataset(
    ds: Dataset,
    *,
    fraction: Optional[float] = None,
    max_samples: Optional[int] = None,
    seed: int,
) -> Dataset:
    """
    Deterministically sample a subset of a dataset.

    - If fraction>=1 and max_samples is None -> returns ds (no-op)
    - If fraction<=0 or max_samples<=0 -> returns empty Subset
    """
    if fraction is None and max_samples is None:
        return ds

    n = len(ds)
    if n <= 0:
        return ds

    k = n
    if fraction is not None:
        f = float(fraction)
        if f <= 0.0:
            return Subset(ds, [])
        if f < 1.0:
            # floor keeps it "smaller/equal" vs rounding up
            k = min(k, max(1, int(n * f)))
    if max_samples is not None:
        m = int(max_samples)
        if m <= 0:
            return Subset(ds, [])
        k = min(k, m)

    if k >= n:
        return ds

    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(n, generator=g).tolist()[:k]
    return Subset(ds, idx)


def _clone_dataloader_with_dataset(dl: DataLoader, dataset: Dataset) -> DataLoader:
    """
    Rebuild a DataLoader for a new dataset, copying common settings.
    """
    kwargs = dict(
        batch_size=getattr(dl, "batch_size", 1),
        shuffle=False,  # never shuffle validation subsets
        num_workers=getattr(dl, "num_workers", 0),
        collate_fn=getattr(dl, "collate_fn", None),
        pin_memory=getattr(dl, "pin_memory", False),
        drop_last=getattr(dl, "drop_last", False),
    )
    # Preserve optional settings if present
    for k in ["persistent_workers", "prefetch_factor", "timeout", "multiprocessing_context", "generator"]:
        if hasattr(dl, k):
            kwargs[k] = getattr(dl, k)
    return DataLoader(dataset, **kwargs)


def _make_val_dataloader_like_train(
    *,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    collate_fn,
    constant_fields: Optional[Mapping[str, Any]] = None,
) -> DataLoader:
    """
    Build a validation DataLoader using MixedDataModule's loader settings.

    Rationale: keeps loader behavior consistent across mixed datasets and avoids
    trying to perfectly "clone" possibly exotic component dataloaders.
    """
    base_collate = collate_fn or default_collate

    def _wrapped_collate(batch):
        out = base_collate(batch)
        if constant_fields and isinstance(out, dict):
            for k, v in constant_fields.items():
                # Force constant fields even if the base collate produced e.g. a
                # list[str] for a per-sample string field. The callback expects a
                # single string dataset name per batch.
                out[k] = v
        return out

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        collate_fn=_wrapped_collate,
    )


class MixedMoleculeImprovementDataModule(pl.LightningDataModule):
    """
    Combine multiple constituent LightningDataModules into a single mixed training dataset
    with static configured fractions.

    - Training: concatenates per-component train datasets (optionally downsampled by fraction).
    - Validation: returns one val dataloader per component (to enable per-dataset metrics cleanly).
    """

    def __init__(
        self,
        *,
        # Accept shared transform_fn for hydra composition convenience.
        # Components should reference it explicitly; MixedMoleculeImprovementDataModule
        # itself does not tokenize.
        transform_fn: Any | None = None,
        # Optional property/distance prompting config.
        # Note: This is *not* used directly by this datamodule, but we accept it so that
        # experiment configs can place prompting under `data.prompting` and we can
        # propagate it to component datamodules (e.g., MoleculeImprovementLightningDataModule),
        # which *do* consume it.
        prompting: Optional[dict] = None,
        components: Mapping[str, pl.LightningDataModule] | Sequence[pl.LightningDataModule],
        train_fractions: Optional[Mapping[str, float]] = None,
        train_max_samples: Optional[Mapping[str, int]] = None,
        val_fractions: Optional[Mapping[str, float]] = None,
        val_max_samples: Optional[Mapping[str, int]] = None,
        seed: int = 0,
        val_seed: Optional[int] = None,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        self._transform_fn = transform_fn
        self._prompting = prompting or {}
        self._component_items = _as_ordered_items(components)
        self._train_fractions = dict(train_fractions) if train_fractions is not None else {}
        self._train_max_samples = dict(train_max_samples) if train_max_samples is not None else {}
        self._val_fractions = dict(val_fractions) if val_fractions is not None else {}
        self._val_max_samples = dict(val_max_samples) if val_max_samples is not None else {}
        self._seed = int(seed)
        self._val_seed = int(val_seed) if val_seed is not None else int(seed)
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._num_workers = int(num_workers)
        self._pin_memory = bool(pin_memory)
        self._drop_last = bool(drop_last)

        self._train_dataset: Optional[Dataset] = None
        self._val_dataloaders: Optional[List[DataLoader]] = None
        self._test_dataloaders: Optional[List[DataLoader]] = None

    @property
    def component_names(self) -> List[str]:
        return [name for name, _dm in self._component_items]

    def prepare_data(self) -> None:
        for _name, dm in self._component_items:
            if hasattr(dm, "prepare_data"):
                dm.prepare_data()

    def setup(self, stage: str = "fit") -> None:
        def _is_disabled(fraction: Optional[float], max_samples: Optional[int]) -> bool:
            """
            Treat explicit 0 (or <=0) as "don't use this dataset".
            This lets configs disable a component without paying setup/load cost.
            """
            try:
                if fraction is not None and float(fraction) <= 0.0:
                    return True
            except Exception:
                pass
            try:
                if max_samples is not None and int(max_samples) <= 0:
                    return True
            except Exception:
                pass
            return False

        # Setup components first (but skip entirely if disabled for this stage)
        for _name, dm in self._component_items:
            # Propagate prompting config (if any) down to component datamodules that support it.
            # This allows experiment configs to specify `data.prompting=...` once at the top level
            # even when `data` is a MixedMoleculeImprovementDataModule.
            try:
                if hasattr(dm, "_prompting"):
                    # If component already has prompting set, don't clobber it unless it's empty.
                    existing = getattr(dm, "_prompting", None)
                    if not existing:
                        setattr(dm, "_prompting", self._prompting)
            except Exception:
                pass

            # If component is a MoleculeImprovementLightningDataModule (or compatible),
            # pass our subset hints down so it can sample while loading (memory efficient).
            try:
                name = _name
                comp_idx = self.component_names.index(name) if name in self.component_names else 0

                tfrac = self._train_fractions.get(name, None)
                vfrac = self._val_fractions.get(name, None)
                tmax = self._train_max_samples.get(name, None)
                vmax = self._val_max_samples.get(name, None)

                # Normalize "full dataset" fractions to None
                if tfrac is not None and float(tfrac) >= 1.0:
                    tfrac = None
                if vfrac is not None and float(vfrac) >= 1.0:
                    vfrac = None

                train_seed = int(self._seed) + 10_000 * int(comp_idx)
                val_seed = int(self._val_seed) + 10_000 * int(comp_idx)

                if hasattr(dm, "_max_pairs_to_load_train"):
                    dm._max_pairs_to_load_train = tmax
                if hasattr(dm, "_max_pairs_to_load_val"):
                    dm._max_pairs_to_load_val = vmax
                if hasattr(dm, "_train_subset_fraction"):
                    dm._train_subset_fraction = tfrac
                if hasattr(dm, "_val_subset_fraction"):
                    dm._val_subset_fraction = vfrac
                if hasattr(dm, "_train_subset_seed"):
                    dm._train_subset_seed = train_seed
                if hasattr(dm, "_val_subset_seed"):
                    dm._val_subset_seed = val_seed

                # If we're subsetting, avoid multi-file "load_all_in_memory" mode.
                if (tmax is not None or vmax is not None or tfrac is not None or vfrac is not None) and hasattr(dm, "_load_all_in_memory"):
                    dm._load_all_in_memory = False
            except Exception:
                pass

            # Decide whether this component is used at all for this stage.
            want_train = (stage in ("fit", "test")) and not _is_disabled(
                self._train_fractions.get(_name, None), self._train_max_samples.get(_name, None)
            )
            want_val = (stage in ("fit", "test", "predict")) and not _is_disabled(
                self._val_fractions.get(_name, None), self._val_max_samples.get(_name, None)
            )
            want_test = (stage in ("test",))  # only if user calls test

            if want_train or want_val or want_test:
                dm.setup(stage)
            else:
                # Fully disabled: don't load it
                continue

        # Build mixed training dataset
        if stage in ("fit", "test"):
            train_parts: List[Dataset] = []
            for name, dm in self._component_items:
                if _is_disabled(self._train_fractions.get(name, None), self._train_max_samples.get(name, None)):
                    continue
                train_dl = dm.train_dataloader()
                ds = train_dl.dataset
                frac = self._train_fractions.get(name, 1.0)
                tmax = self._train_max_samples.get(name, None)
                # Prefer dataset-internal sampling when available (avoids materializing all pairs).
                if getattr(ds, "_subset_applied", False):
                    ds_sub = ds
                else:
                    ds_sub = _subset_dataset(
                        ds,
                        fraction=frac,
                        max_samples=tmax,
                        seed=(self._seed + 10_000 * int(self.component_names.index(name) if name in self.component_names else 0)),
                    )
                ds_sub = _AddConstantFieldsDataset(ds_sub, fields={"dataset_name": name})
                train_parts.append(ds_sub)
            if not train_parts:
                # No enabled training datasets; keep an empty dataset
                self._train_dataset = ConcatDataset([])
            else:
                self._train_dataset = ConcatDataset(train_parts)

            # Keep per-component val/test dataloaders (no mixing) for clean metrics separation
            self._val_dataloaders = []
            for name, dm in self._component_items:
                if _is_disabled(self._val_fractions.get(name, None), self._val_max_samples.get(name, None)):
                    continue
                dl = dm.val_dataloader()
                ds = dl.dataset
                vfrac = self._val_fractions.get(name, 1.0)
                vmax = self._val_max_samples.get(name, None)
                if getattr(ds, "_subset_applied", False):
                    ds_sub = ds
                else:
                    ds_sub = _subset_dataset(
                        ds,
                        fraction=vfrac,
                        max_samples=vmax,
                        seed=(self._val_seed + 10_000 * int(self.component_names.index(name) if name in self.component_names else 0)),
                    )
                ds_sub = _AddConstantFieldsDataset(ds_sub, fields={"dataset_name": name})
                self._val_dataloaders.append(
                    _make_val_dataloader_like_train(
                        dataset=ds_sub,
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        pin_memory=self._pin_memory,
                        collate_fn=getattr(dl, "collate_fn", None),
                        constant_fields={"dataset_name": name},
                    )
                )
            try:
                self._test_dataloaders = [dm.test_dataloader() for _name, dm in self._component_items]
            except Exception:
                self._test_dataloaders = None

        if stage == "predict":
            # Prediction: just delegate to components (list of loaders)
            self._val_dataloaders = [dm.val_dataloader() for _name, dm in self._component_items]

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise ValueError("Train dataset not initialized. Call setup() first.")
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )

    def val_dataloader(self):
        if self._val_dataloaders is None:
            raise ValueError("Validation dataloaders not initialized. Call setup() first.")
        return self._val_dataloaders

    def test_dataloader(self):
        # Allow returning None if components don't have test loaders.
        return self._test_dataloaders


