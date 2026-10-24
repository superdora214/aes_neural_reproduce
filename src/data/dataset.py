"""
AES 差分-神经区分器的训练数据生成
====================================
对照张柳2024 (IEICE 2024) 第 2.3 / 3.1 节：

  - 每条样本用**独立的随机密钥** K_i 和随机明文 P0；
  - 标签 Y=1（真实对）：P1 = P0 XOR Δ，Δ = 0x80（第 0 号字节的最高位）；
  - 标签 Y=0（随机对）：P1 换成一条新的随机明文；
  - 两条明文加密 nr 轮后，从密文里取指定的 nbytes 个字节；
  - 编码成位向量喂给网络：通道 = (密文侧, 该侧的字节序号)，每通道 8 个比特位（MSB 在前）。

张柳2024 的关键结论正是"取哪几个字节"决定成败：
  论文单字节设定的准确率接近随机（Acc ≈ 0.51）；
  取列内相邻的 2 个字节 {0,1} → 2 轮区分器 Acc ≈ 0.998。
"""
import torch

from src.aes.aes import aes_encrypt

# 输入差分：第 0 号字节的最高位（张柳2024 用 0x80）
DEFAULT_DIFF = (0, 0x80)


def encode_bits(c_side0, c_side1, nbytes):
    """
    把两侧密文各 nbytes 个字节编码成位向量。
    c_side0/c_side1: [N, nbytes] int64
    返回 [N, 2*nbytes*8] uint8，MSB 在前。
    """
    n = c_side0.shape[0]
    x = torch.zeros((n, 2 * nbytes * 8), dtype=torch.uint8, device=c_side0.device)
    for b in range(nbytes):
        for i in range(8):
            x[:, b * 8 + i] = (c_side0[:, b] >> (7 - i)) & 1
            x[:, nbytes * 8 + b * 8 + i] = (c_side1[:, b] >> (7 - i)) & 1
    return x


def make_aes_train_data(n, nr=2, byte_idx=(0, 1), diff=DEFAULT_DIFF,
                        omit_final_mc=False, device="cuda", chunk=200000, seed=None):
    """
    生成 n 条样本。
      nr             : 轮数（张柳2024 的 2 轮区分器用 nr=2，且末轮含列混合）
      byte_idx       : 从密文里取哪些字节。默认 (0, 1) = 第 0 列的上两个字节
      omit_final_mc  : 末轮是否省略列混合（3 轮"纯区分攻击"用 True）
    返回 (X[N, 2*len(byte_idx)*8] uint8, Y[N] uint8)
    """
    if seed is not None:
        torch.manual_seed(seed)
    nbytes = len(byte_idx)
    idx = torch.tensor(list(byte_idx), dtype=torch.long, device=device)

    xs, ys, done = [], [], 0
    while done < n:
        m = min(chunk, n - done)
        y = torch.randint(0, 2, (m,), device=device)
        key = torch.randint(0, 256, (m, 16), dtype=torch.int64, device=device)
        p0 = torch.randint(0, 256, (m, 16), dtype=torch.int64, device=device)

        p1 = p0.clone()
        p1[:, diff[0]] ^= diff[1]
        neg = y == 0
        if neg.any():
            p1[neg] = torch.randint(0, 256, (int(neg.sum()), 16),
                                    dtype=torch.int64, device=device)

        c0 = aes_encrypt(p0, key, nr=nr, omit_final_mc=omit_final_mc)
        c1 = aes_encrypt(p1, key, nr=nr, omit_final_mc=omit_final_mc)

        xs.append(encode_bits(c0[:, idx], c1[:, idx], nbytes))
        ys.append(y)
        done += m

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    X, Y = make_aes_train_data(1000, nr=2, device=device)
    print(f"X: {tuple(X.shape)}, Y: {tuple(Y.shape)}, 标签1占比: {Y.float().mean():.3f}")
    assert X.shape == (1000, 32) and X.max() <= 1

    t0 = time.time()
    make_aes_train_data(1_000_000, nr=2, device=device)
    print(f"生成 100 万条 2 轮样本耗时: {time.time() - t0:.2f}s")
