"""Post-training checks: independent classification and paired subkey recovery."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from key_recovery import check_round_boundary, score_candidates
from src.aes.aes import aes_encrypt, expand_key, xtime
from src.data.dataset import make_aes_train_data
from src.model.distinguisher import make_distinguisher


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def wilson(successes, total):
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, min(rate, center - half)), min(1.0, max(rate, center + half))]


def classical_rule(x):
    # Layout: C[0], C[1], C'[0], C'[1], each encoded as eight MSB-first bits.
    weights = 2 ** torch.arange(7, -1, -1, device=x.device)
    values = (x.long().reshape(-1, 4, 8) * weights).sum(dim=2)
    d0, d1 = values[:, 0] ^ values[:, 2], values[:, 1] ^ values[:, 3]
    return (d1 != 0) & (d0 == xtime(d1))


def metrics(predicted, labels):
    positive = labels.bool()
    tp = int((predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    return {"samples": len(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(labels), "tpr": tp / (tp + fn),
            "tnr": tn / (tn + fp)}


@torch.inference_mode()
def classify(model, x, labels):
    probabilities = torch.cat([model(batch).flatten() for batch in x.split(10000)])
    assert torch.isfinite(probabilities).all()
    return probabilities > 0.5, {
        **metrics(probabilities > 0.5, labels),
        "mse": float(((probabilities - labels.float()) ** 2).mean()),
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backup-run", type=Path, help="Optional original full-run folder for epoch-14/15 checks")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert args.trials > 0 and args.samples > 0
    args.outdir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    check_round_boundary(device)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model = make_distinguisher(depth=5, num_blocks=2, word_size=8).to(device).eval()
    best_state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    epoch1_equal = None
    training_seed = None
    if args.backup_run:
        config = json.loads((args.backup_run / "config.json").read_text())
        recorded = json.loads((args.backup_run / "test_metrics.json").read_text())
        assert checkpoint_hash == recorded["checkpoint_sha256"]
        epoch1 = torch.load(args.backup_run / "net_r2b01_epoch001.pt", map_location=device, weights_only=True)
        epoch1_equal = best_state.keys() == epoch1.keys() and all(torch.equal(best_state[n], epoch1[n]) for n in best_state)
        assert epoch1_equal
        training_seed = config["arguments"]["seed"]
    model.load_state_dict(best_state)
    metadata = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "classification_seed": 20261029, "attack_seed": 20261030,
                "trials": args.trials, "max_pairs": 3, "candidate_batch": 4096,
                "best_state_equals_epoch1_tensors": epoch1_equal,
                "original_training_seed": training_seed}
    write_json(args.outdir / "metadata.json", metadata)

    # Fresh samples; every classifier/checkpoint sees the exact same inputs.
    x, labels = make_aes_train_data(args.samples, nr=2, byte_idx=(0, 1),
                                   device=str(device), seed=20261029)
    analytic = classical_rule(x)
    assert bool(analytic[labels.bool()].all()), "Two-round differential relation failed."
    pred, neural = classify(model, x, labels)
    evaluation = {"seed": 20261029, "samples": args.samples, "neural_best": neural,
                  "classical_rule": metrics(analytic, labels),
                  "neural_rule_disagreements": int((pred != analytic).sum()),
                  "uniform_negative_reference_fpr": 255 / 65536,
                  "uniform_negative_reference_tnr": 1 - 255 / 65536,
                  "balanced_uniform_reference_accuracy": 1 - 255 / (2 * 65536)}
    # Enumerate all 16-bit differences: exactly 255 satisfy the nonzero relation.
    all_values = torch.arange(65536, device=device)
    assert int((((all_values & 255) != 0) & ((all_values >> 8) == xtime(all_values & 255))).sum()) == 255
    for epoch in ((14, 15) if args.backup_run else ()):
        state = torch.load(args.backup_run / f"net_r2b01_epoch{epoch:03d}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        _, evaluation[f"epoch_{epoch}"] = classify(model, x, labels)
    model.load_state_dict(best_state)
    write_json(args.outdir / "independent_classification.json", evaluation)
    print("[classification] " + json.dumps(evaluation), flush=True)
    del x, labels, analytic, pred

    # Pre-generate the same keys and three ordered pairs for every pair-count arm.
    torch.manual_seed(20261030)
    keys = torch.randint(256, (args.trials, 16), device=device)
    p0 = torch.randint(256, (args.trials, 3, 16), device=device)
    p1 = p0.clone()
    p1[:, :, 0] ^= 0x80
    expanded_keys = keys[:, None, :].expand(-1, 3, -1).reshape(-1, 16)
    c0 = aes_encrypt(p0.reshape(-1, 16), expanded_keys, nr=3, omit_final_mc=True).reshape(-1, 3, 16)
    c1 = aes_encrypt(p1.reshape(-1, 16), expanded_keys, nr=3, omit_final_mc=True).reshape(-1, 3, 16)
    torch.save({"keys": keys.cpu(), "p0": p0.cpu(), "p1": p1.cpu()}, args.outdir / "simulation_inputs.pt")
    # The saved keys belong to this local simulation only; the scorer never gets them.
    round_keys = expand_key(keys, nr=3)[:, 3]
    trials = []
    loop_start = time.perf_counter()
    for trial in range(args.trials):
        for pairs in (1, 2, 3):
            sync(device)
            started = time.perf_counter()
            scores = score_candidates(model, c0[trial, :pairs], c1[trial, :pairs], 4096)
            sync(device)
            seconds = time.perf_counter() - started
            assert torch.isfinite(scores).all()
            truth = (int(round_keys[trial, 0]) << 8) | int(round_keys[trial, 13])
            greater = int((scores > scores[truth]).sum())
            equal = int((scores == scores[truth]).sum())
            guess = int(scores.argmax())
            trials.append({"trial": trial + 1, "pairs": pairs, "true_subkey": f"{truth:04x}",
                           "guess": f"{guess:04x}", "top1_correct": guess == truth,
                           "unique_rank1": greater == 0 and equal == 1,
                           "rank_best": greater + 1, "rank_worst": greater + equal,
                           "ties_with_true": equal, "scoring_seconds": seconds})
        if (trial + 1) % 50 == 0 or trial + 1 == args.trials:
            write_json(args.outdir / "attack_trials.json", trials)
            print(f"[attack] {trial + 1}/{args.trials} paired keys, elapsed={time.perf_counter() - loop_start:.1f}s", flush=True)
    summary = []
    for pairs in (1, 2, 3):
        arm = [row for row in trials if row["pairs"] == pairs]
        top1 = sum(row["top1_correct"] for row in arm)
        unique = sum(row["unique_rank1"] for row in arm)
        summary.append({"pairs": pairs, "encryption_queries": pairs * 2, "trials": args.trials,
                        "top1_correct": top1, "top1_rate": top1 / args.trials,
                        "top1_wilson_95": wilson(top1, args.trials),
                        "unique_rank1": unique, "unique_rate": unique / args.trials,
                        "unique_wilson_95": wilson(unique, args.trials),
                        "true_in_top_tie": sum(row["rank_best"] == 1 for row in arm),
                        "max_rank_worst": max(row["rank_worst"] for row in arm),
                        "median_rank_best": statistics.median(row["rank_best"] for row in arm),
                        "median_scoring_seconds": statistics.median(row["scoring_seconds"] for row in arm)})
    write_json(args.outdir / "attack_summary.json", summary)
    with (args.outdir / "attack_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    print("[summary] " + json.dumps(summary), flush=True)
    print("[done] " + str(args.outdir), flush=True)


if __name__ == "__main__":
    main()
