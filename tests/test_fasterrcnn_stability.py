"""Regression checks for FP32 detection and synchronized non-finite failures."""
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import fasterrcnn_train as trainer
from common import CocoDataset


def _bad_loss_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=0.005)
        target = {"image_id": torch.tensor(100 + rank),
                  "boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64)}
        # Only rank 1 fails, but rank 0 must also exit before backward.
        with pytest.raises(RuntimeError, match="Non-finite loss"):
            trainer.require_finite({"loss_classifier": torch.tensor(float("nan") if rank else 1.)},
                output=output, rank=rank, epoch=0, step=841, targets=[target],
                optimizer=optimizer, amp=False, stage="loss")
    finally:
        dist.destroy_process_group()


def test_one_rank_nonfinite_stops_both_and_records_batch(tmp_path):
    mp.spawn(_bad_loss_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)),
             nprocs=2, join=True)
    reports = [json.loads((tmp_path / f"nonfinite_rank{rank}.json").read_text()) for rank in range(2)]
    assert reports[0]["values"]["loss_classifier"] == 1.
    assert reports[1]["values"]["loss_classifier"] == "nan"
    assert [r["targets"][0]["image_id"] for r in reports] == [100, 101]
    assert all(r["step"] == 841 and r["epoch"] == 1 and not r["amp"] for r in reports)


@pytest.mark.parametrize("fail", [False, True])
def test_training_always_closes_process_group(monkeypatch, fail):
    closed = []
    def run(config):
        if fail:
            raise RuntimeError("training failure")
    monkeypatch.setattr(trainer, "_train", run)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "destroy_process_group", lambda: closed.append(True))
    if fail:
        with pytest.raises(RuntimeError, match="training failure"):
            trainer.train({})
    else:
        trainer.train({})
    assert closed == [True]


def test_fp32_real_positive_and_all_negative_batches(tmp_path):
    root = Path(__file__).resolve().parents[1] / "dataset/varroa_ready"
    if not (root / "images/train").is_dir():
        pytest.skip("Prepared local dataset unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dataset = CocoDataset(root, "train", 128)
    pos = next(i for i, im in enumerate(dataset.images) if dataset.annotations[im["id"]])
    neg = next(i for i, im in enumerate(dataset.images) if not dataset.annotations[im["id"]])
    model = trainer.build_model(pretrained=False, size=128).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00005)
    for step, indices in enumerate(([pos, neg], [neg, neg])):
        samples = [dataset[i] for i in indices]
        images, targets = [s[0] for s in samples], [s[1] for s in samples]
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        args = dict(output=tmp_path, rank=0, epoch=0, step=step, targets=targets,
                    optimizer=optimizer, amp=False)
        trainer.require_finite(losses, stage="loss", **args)
        sum(losses.values()).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        trainer.require_finite({"grad_norm": norm}, stage="gradient", **args)
        optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_kaggle_notebook_uses_fp32_and_current_trainer():
    lab = Path(__file__).resolve().parents[1]
    notebook = json.loads((lab / "notebooks/train_fasterrcnn_kaggle.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"]]
    assert any("AMP = False" in s for s in sources)
    source = (lab / "training/fasterrcnn_train.py").read_text()
    assert any(s.startswith("%%writefile ") and s.endswith(source) for s in sources)
