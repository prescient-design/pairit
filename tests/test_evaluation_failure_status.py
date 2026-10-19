"""A callback failure must not turn into a successful evaluation command."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


class BrokenCallback:
    def evaluate(self, model, dataloader):
        raise RuntimeError("sampling failed")


def test_evaluation_raises_after_writing_failure_report(tmp_path, monkeypatch):
    module = importlib.import_module("pairit.cmdline.evaluate")
    objects = {
        "callback": BrokenCallback(),
        "model": torch.nn.Linear(1, 1),
        "data": SimpleNamespace(prepare_data=lambda: None, setup=lambda: None,
                                test_dataloader=lambda: []),
    }
    monkeypatch.setattr(module.hydra.utils, "instantiate", lambda cfg: objects[cfg.kind])
    cfg = OmegaConf.create({
        "model": {"kind": "model", "ckpt_path": None},
        "datamodule": {"kind": "data"},
        "callbacks": [{"kind": "callback"}],
        "split": "test", "output_dir": str(tmp_path),
    })
    with pytest.raises(RuntimeError, match="Evaluation failed: BrokenCallback: sampling failed"):
        module.evaluate.__wrapped__(cfg)
    assert "sampling failed" in (tmp_path / "evaluation_report.md").read_text()
