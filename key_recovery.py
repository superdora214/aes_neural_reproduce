"""Enumerate a 16-bit final-round subkey of 3-round AES using a 2-round ND."""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from src.aes.aes import SBOX, aes_encrypt, expand_key
from src.data.dataset import encode_bits
from src.model.distinguisher import make_distinguisher


def inverse_sbox(device):
    table = torch.empty(256, dtype=torch.long, device=device)
    table[torch.tensor(SBOX, device=device)] = torch.arange(256, device=device)
    return table


def ciphertext_positions(state_positions):
    # Forward ShiftRows moves source (row, col) to (row, col - row).
    return tuple(4 * ((i // 4 - i % 4) % 4) + i % 4 for i in state_positions)


def check_round_boundary(device):
    torch.manual_seed(7421)
    p = torch.randint(256, (16, 16), device=device)
    key = torch.randint(256, (16, 16), device=device)
    c = aes_encrypt(p, key, nr=3, omit_final_mc=True)
    k3 = expand_key(key, nr=3)[:, 3]
    positions = ciphertext_positions(range(16))
    recovered = inverse_sbox(device)[c[:, positions] ^ k3[:, positions]]
    expected = aes_encrypt(p, key, nr=2, omit_final_mc=False)
    assert torch.equal(recovered, expected), "Last-round stripping or byte mapping is wrong."
    assert ciphertext_positions((0, 1)) == (0, 13)


@torch.inference_mode()
def score_candidates(model, c0, c1, candidate_batch):
    """Uses only ciphertexts and a model, never the real key or intermediate states."""
    device = c0.device
    inv = inverse_sbox(device)
    positions = ciphertext_positions((0, 1))
    a, b = c0[:, positions], c1[:, positions]
    pairs = len(c0)
    scores = torch.empty(65536, dtype=torch.float64, device=device)
    for start in range(0, 65536, candidate_batch):
        values = torch.arange(start, min(start + candidate_batch, 65536), device=device)
        guessed = torch.stack((values >> 8, values & 255), dim=1)
        stripped_a = inv[a[None, :, :] ^ guessed[:, None, :]].reshape(-1, 2)
        stripped_b = inv[b[None, :, :] ^ guessed[:, None, :]].reshape(-1, 2)
        x = encode_bits(stripped_a, stripped_b, 2)
        probability = model(x).flatten().double().clamp(1e-6, 1 - 1e-6)
        logits = torch.log(probability) - torch.log1p(-probability)
        scores[start:start + len(values)] = logits.reshape(-1, pairs).sum(dim=1)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--candidate-batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path)
    args = ap.parse_args()
    if min(args.trials, args.pairs, args.candidate_batch) < 1:
        ap.error("trials, pairs, and candidate-batch must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA is unavailable in this Python environment")
    if not args.checkpoint.is_file():
        ap.error(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    check_round_boundary(device)
    print("[check] Last-round stripping matches all 16 bytes of the 2-round state.", flush=True)
    print("[mapping] ND state bytes [0,1] <- ciphertext/key bytes [0,13].", flush=True)
    model = make_distinguisher(depth=5, num_blocks=2, word_size=8)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    torch.manual_seed(args.seed)
    outdir = args.outdir or Path("runs") / datetime.now().strftime("key_recovery_%Y%m%d_%H%M%S_%f")
    outdir.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "attack_rounds": 3, "attack_omit_final_mc": True,
        "distinguisher_rounds": 2, "distinguisher_omit_final_mc": False,
        "distinguisher_bytes": [0, 1], "ciphertext_key_bytes": [0, 13],
        "recovered_bits": 16, "candidates": 65536,
        "pairs_per_trial": args.pairs, "encryption_queries_per_trial": 2 * args.pairs,
        "seed": args.seed, "candidate_batch": args.candidate_batch,
        "score": "sum log(p/(1-p)), clamp probabilities to [1e-6,1-1e-6]",
        "trials": [],
    }
    report_path = outdir / "results.json"
    start_all = time.perf_counter()
    for trial in range(args.trials):
        started = time.perf_counter()
        # The key stays inside the simulated encryption oracle and evaluation.
        real_key = torch.randint(256, (1, 16), device=device)
        p0 = torch.randint(256, (args.pairs, 16), device=device)
        p1 = p0.clone()
        p1[:, 0] ^= 0x80
        key_batch = real_key.expand(args.pairs, -1)
        c0 = aes_encrypt(p0, key_batch, nr=3, omit_final_mc=True)
        c1 = aes_encrypt(p1, key_batch, nr=3, omit_final_mc=True)
        scores = score_candidates(model, c0, c1, args.candidate_batch)

        # Ground truth is consulted only after all candidate scores are computed.
        k3 = expand_key(real_key, nr=3)[0, 3]
        truth = (int(k3[0]) << 8) | int(k3[13])
        best = int(scores.argmax())
        true_score = float(scores[truth])
        greater = int((scores > scores[truth]).sum())
        equal = int((scores == scores[truth]).sum())
        entry = {
            "trial": trial + 1, "true_subkey": f"{truth:04x}",
            "guessed_subkey": f"{best:04x}", "top1_correct": best == truth,
            "rank_best": greater + 1, "rank_worst": greater + equal,
            "tied_with_true": equal, "unique_rank1": greater == 0 and equal == 1,
            "true_score": true_score, "top_score": float(scores[best]),
            "seconds": round(time.perf_counter() - started, 4),
        }
        report["trials"].append(entry)
        print(f"[{trial + 1}/{args.trials}] true={truth:04x} guess={best:04x} "
              f"rank={greater + 1}..{greater + equal} "
              f"correct={best == truth} seconds={entry['seconds']:.2f}", flush=True)
        # Save each completed trial so a failed or interrupted run leaves evidence.
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["summary"] = {
        "trials": args.trials,
        "top1_correct": sum(t["top1_correct"] for t in report["trials"]),
        "unique_rank1": sum(t["unique_rank1"] for t in report["trials"]),
        "true_in_top_tie": sum(t["rank_best"] == 1 for t in report["trials"]),
        "seconds": round(time.perf_counter() - start_all, 3),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[summary] " + json.dumps(report["summary"]), flush=True)
    print(f"[saved] {report_path}", flush=True)
    print("[scope] This recovers only 16 bits of K3, not the full AES master key.", flush=True)


if __name__ == "__main__":
    main()
