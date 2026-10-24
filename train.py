"""
张柳2024《AES 上的差分-神经密码分析》—— 区分器训练脚本
==========================================================
用法示例：
  python train.py --rounds 2 --bytes 0,1 --train-size 2000000 --epochs 12
  python train.py --rounds 3 --bytes 0,1 --train-size 2000000 --epochs 12 --omit-final-mc

对照论文：
  - 表 1  ：2 轮完整密文 Acc 0.5890 / TPR 1.0 / TNR 0.1780
  - 表 5  ：2 轮两字节   Acc 0.9980 / TPR 1.0 / TNR 0.9961   ← 本脚本默认复现这条
  - 表 3  ：3 轮两字节   TNR 约 1/128，无法用于密钥筛选
训练配置照论文：MSE 损失 + L2 + Adam + 循环学习率（周期 n=9，α=1e-4，β=2e-3）。
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import make_aes_train_data  # noqa: E402
from src.model.distinguisher import make_distinguisher  # noqa: E402


def cyclic_lr(period, high=0.002, low=0.0001):
    """
    论文的 l_i = α + ((n-i) mod (n+1))/n · (β-α)，即三角循环学习率。
    ⚠️ LambdaLR 是把返回值**乘**到优化器的初始 lr 上，所以这里返回倍率而不是绝对值。
    """
    def fn(epoch):
        lr = low + ((period - epoch) % (period + 1)) / period * (high - low)
        return lr / high
    return fn


@torch.no_grad()
def evaluate(model, X, Y, batch=10000):
    model.eval()
    preds = []
    for i in range(0, X.shape[0], batch):
        preds.append(model(X[i:i + batch]))
    p = torch.cat(preds).view(-1)
    y = Y.float().view(-1)
    loss = torch.nn.functional.mse_loss(p, y).item()
    hard = (p > 0.5).float()
    acc = (hard == y).float().mean().item()
    pos, neg = y == 1, y == 0
    tpr = (hard[pos] == 1).float().mean().item() if pos.any() else float("nan")
    tnr = (hard[neg] == 0).float().mean().item() if neg.any() else float("nan")
    model.train()
    return acc, tpr, tnr, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--bytes", type=str, default="0,1",
                    help="从密文里取哪些字节，如 '0,1'（列内相邻）或 '0'（单字节）")
    ap.add_argument("--train-size", type=int, default=2_000_000)
    ap.add_argument("--eval-size", type=int, default=100_000)
    ap.add_argument("--test-size", type=int, default=0,
                    help="最终独立测试集大小；0 表示不执行最终测试")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--period", type=int, default=9,
                    help="论文学习率公式中的 n，实际循环长度为 n+1")
    ap.add_argument("--lr-high", type=float, default=0.002)
    ap.add_argument("--lr-low", type=float, default=0.0001)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--omit-final-mc", action="store_true",
                    help="末轮省略列混合（论文的 3 轮'纯区分攻击'设定）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every-epoch", action="store_true")
    args = ap.parse_args()

    if min(args.train_size, args.batch_size) < 2:
        ap.error("train-size and batch-size must be at least 2")
    if min(args.eval_size, args.epochs, args.period) < 1 or args.test_size < 0:
        ap.error("eval-size, epochs and period must be positive; test-size must be nonnegative")

    byte_idx = tuple(int(b) for b in args.bytes.split(","))
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"r{args.rounds}b{''.join(map(str, byte_idx))}" + ("_nomc" if args.omit_final_mc else "")
    output_dir = Path(args.outdir)
    best_path = output_dir / f"net_{tag}.pt"
    config = {
        "arguments": vars(args),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "seeds": {"train": args.seed, "validation": args.seed + 1,
                  "test": args.seed + 2, "model": args.seed + 3},
        "learning_rate_formula": "low + ((n-epoch) % (n+1))/n * (high-low)",
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__).resolve(),
                         Path(__file__).resolve().parent / "src/data/dataset.py",
                         Path(__file__).resolve().parent / "src/model/distinguisher.py",
                         Path(__file__).resolve().parent / "src/aes/aes.py"]
        },
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[配置] 轮数={args.rounds} 取字节={byte_idx} 末轮省略MC={args.omit_final_mc}")
    print(f"       训练={args.train_size} 验证={args.eval_size} epoch={args.epochs} "
          f"depth={args.depth} 设备={args.device}")

    t0 = time.time()
    Xtr, Ytr = make_aes_train_data(
        args.train_size, nr=args.rounds, byte_idx=byte_idx,
        omit_final_mc=args.omit_final_mc, device=args.device, seed=args.seed)
    Xva, Yva = make_aes_train_data(
        args.eval_size, nr=args.rounds, byte_idx=byte_idx,
        omit_final_mc=args.omit_final_mc, device=args.device, seed=args.seed + 1)
    print(f"[数据] 生成耗时 {time.time() - t0:.1f}s   "
          f"Xtr={tuple(Xtr.shape)} Xva={tuple(Xva.shape)}")

    # 输入编码 = (2 个密文 × nbytes 个字节) 个通道，每通道 8 个比特位
    torch.manual_seed(args.seed + 3)
    model = make_distinguisher(depth=args.depth, num_blocks=len(byte_idx),
                               word_size=8).to(args.device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[网络] 参数量 {nparam:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_high, weight_decay=args.l2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cyclic_lr(args.period, args.lr_high, args.lr_low))
    lossf = nn.MSELoss()

    n = Xtr.shape[0]
    best = None
    history = []
    for ep in range(1, args.epochs + 1):
        epoch_lr = opt.param_groups[0]['lr']
        perm = torch.randperm(n, device=args.device)
        te = time.time()
        tot, cnt = 0.0, 0
        for i in range(0, n, args.batch_size):
            bi = perm[i:i + args.batch_size]
            if bi.numel() < 2:
                continue
            xb = Xtr[bi].float()
            yb = Ytr[bi].float().view(-1, 1)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * bi.numel()
            cnt += bi.numel()
        sched.step()
        acc, tpr, tnr, vloss = evaluate(model, Xva, Yva)
        history.append({"epoch": ep, "lr": epoch_lr, "train_loss": tot / max(cnt, 1),
                        "val_loss": vloss, "val_acc": acc, "tpr": tpr, "tnr": tnr,
                        "seconds": time.time() - te})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"  epoch {ep:2d}/{args.epochs}  lr={epoch_lr:.5f}  "
              f"train_loss={tot / max(cnt, 1):.5f}  val_loss={vloss:.5f}  "
              f"val_acc={acc:.4f}  TPR={tpr:.4f}  TNR={tnr:.4f}  ({time.time() - te:.1f}s)")
        # 论文的做法：取**验证损失最低**的那个 epoch，而不是挑准确率/TNR 最高的
        if best is None or vloss < best[3]:
            best = (acc, tpr, tnr, vloss, ep)
            torch.save(model.state_dict(), best_path)
        if args.save_every_epoch:
            torch.save(model.state_dict(), output_dir / f"net_{tag}_epoch{ep:03d}.pt")

    print(f"\n[最佳] epoch {best[4]}（验证损失最低）  Acc={best[0]:.4f}  "
          f"TPR={best[1]:.4f}  TNR={best[2]:.4f}")
    if best[2] > 0:
        print(f"       TNR 的倒数 ≈ 1/{1 / best[2]:.1f}")
    print("[论文对照] 2 轮两字节 Acc 0.9980/TNR 0.9961；2 轮单字节 Acc 0.5136/TNR 0.4628；"
          "3 轮两字节 TNR ≈ 1/128")
    print(f"[模型] {best_path}")
    if args.test_size:
        del Xtr, Ytr, Xva, Yva
        model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
        Xtest, Ytest = make_aes_train_data(
            args.test_size, nr=args.rounds, byte_idx=byte_idx,
            omit_final_mc=args.omit_final_mc, device=args.device, seed=args.seed + 2)
        acc, tpr, tnr, loss = evaluate(model, Xtest, Ytest)
        result = {"selected_epoch": best[4], "samples": args.test_size,
                  "seed": args.seed + 2, "accuracy": acc, "tpr": tpr,
                  "tnr": tnr, "mse_loss": loss,
                  "checkpoint": best_path.name,
                  "checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest()}
        (output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[独立测试] Acc={acc:.6f} TPR={tpr:.6f} TNR={tnr:.6f} MSE={loss:.6f}")


if __name__ == "__main__":
    main()
