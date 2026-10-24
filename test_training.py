"""Check the paper LR schedule and the final held-out evaluation pipeline."""
import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.data.dataset import make_aes_train_data
from src.model.distinguisher import make_distinguisher
from train import cyclic_lr, evaluate


def main():
    root = Path(__file__).resolve().parent
    schedule = cyclic_lr(9)
    assert math.isclose(schedule(0) * 0.002, 0.002)
    assert math.isclose(schedule(9) * 0.002, 0.0001)
    assert math.isclose(schedule(10) * 0.002, 0.002)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (root / "results").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="training-check-", dir=root / "results") as folder:
        subprocess.run([
            sys.executable, str(root / "train.py"),
            "--train-size", "2000", "--eval-size", "400", "--test-size", "400",
            "--epochs", "2", "--batch-size", "200", "--seed", "7",
            "--device", device, "--save-every-epoch", "--outdir", folder,
        ], check=True, cwd=root)
        folder = Path(folder)
        history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
        result = json.loads((folder / "test_metrics.json").read_text(encoding="utf-8"))
        assert len(history) == 2 and result["seed"] == 9
        assert result["selected_epoch"] == min(history, key=lambda row: row["val_loss"])["epoch"]
        best = torch.load(folder / "net_r2b01.pt", weights_only=True, map_location="cpu")
        selected = torch.load(folder / f"net_r2b01_epoch{result['selected_epoch']:03d}.pt",
                              weights_only=True, map_location="cpu")
        assert all(torch.equal(best[name], selected[name]) for name in best)
        model = make_distinguisher(depth=5, num_blocks=2, word_size=8).to(device)
        model.load_state_dict(best)
        x, y = make_aes_train_data(400, device=device, seed=9)
        metrics = evaluate(model, x, y)
        for actual, name in zip(metrics, ["accuracy", "tpr", "tnr", "mse_loss"]):
            assert math.isclose(actual, result[name], abs_tol=1e-7)
    print("PASS: paper LR schedule, epoch selection, saved weights, independent test.")


if __name__ == "__main__":
    main()
