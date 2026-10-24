"""
AES 实现正确性校验
==================
1) FIPS-197 附录 C.1 官方测试向量（完整 10 轮 AES-128）；
2) 与一份独立的标量参考实现逐轮比对（覆盖 nr=1..10 的每一个中间状态）。

第二项是关键：如果只测完整 10 轮，缩减轮（张柳2024 用 2 轮）的索引/字节序错误
可能测不出来。这里把每一轮的输出都对齐比一遍。
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.aes.aes import SBOX, aes_encrypt  # noqa: E402


# ---------------- 独立标量参考实现（用列表和循环，不走张量） ----------------

def _xt(a):
    a <<= 1
    if a & 0x100:
        a = (a ^ 0x11B) & 0xFF
    return a


def _mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = _xt(a)
        b >>= 1
    return r


def ref_expand_key(key, nr):
    w = [list(key[4 * i:4 * i + 4]) for i in range(4)]
    rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
    for i in range(4, 4 * (nr + 1)):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = t[1:] + t[:1]
            t = [SBOX[x] for x in t]
            t[0] ^= rcon[i // 4 - 1]
        w.append([w[i - 4][j] ^ t[j] for j in range(4)])
    # 第 r 轮密钥 = 连续 16 字节
    return [sum(w[4 * r:4 * r + 4], []) for r in range(nr + 1)]


def ref_round(s, rk, last):
    """last=True 表示该轮省略列混合（完整 AES 的末轮）"""
    s = [SBOX[x] for x in s]
    # 行移位：列优先，第 r 行左移 r 个字节
    out = [0] * 16
    for r in range(4):
        for c in range(4):
            out[4 * c + r] = s[4 * ((c + r) % 4) + r]
    s = out
    if not last:
        out = [0] * 16
        for c in range(4):
            col = s[4 * c:4 * c + 4]
            out[4 * c + 0] = _mul(col[0], 2) ^ _mul(col[1], 3) ^ col[2] ^ col[3]
            out[4 * c + 1] = col[0] ^ _mul(col[1], 2) ^ _mul(col[2], 3) ^ col[3]
            out[4 * c + 2] = col[0] ^ col[1] ^ _mul(col[2], 2) ^ _mul(col[3], 3)
            out[4 * c + 3] = _mul(col[0], 3) ^ col[1] ^ col[2] ^ _mul(col[3], 2)
        s = out
    return [a ^ b for a, b in zip(s, rk)]


def ref_encrypt(pt, key, nr, omit_final_mc):
    rks = ref_expand_key(key, nr)
    s = [a ^ b for a, b in zip(pt, rks[0])]
    for r in range(1, nr + 1):
        s = ref_round(s, rks[r], last=(omit_final_mc and r == nr))
    return s


# ---------------- 测试 ----------------

def test_fips197():
    key = list(range(16))                                   # 000102...0f
    pt = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
          0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]
    expect = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    got = aes_encrypt(
        torch.tensor([pt], dtype=torch.int64, device=device),
        torch.tensor([key], dtype=torch.int64, device=device),
        nr=10, omit_final_mc=True,
    )[0].cpu().tolist()
    got = bytes(got)
    print(f"[1] FIPS-197 完整 AES-128: got={got.hex()} expect={expect.hex()}")
    assert got == expect, "FIPS-197 测试向量不匹配！"
    print("    ✓ 通过")


def test_round_by_round():
    """逐轮比对张量实现与标量参考实现（nr=1..10，两种末轮设定都测）"""
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 256
    pt_l = [[random.randrange(256) for _ in range(16)] for _ in range(n)]
    k_l = [[random.randrange(256) for _ in range(16)] for _ in range(n)]
    pt = torch.tensor(pt_l, dtype=torch.int64, device=device)
    key = torch.tensor(k_l, dtype=torch.int64, device=device)

    for nr in range(1, 11):
        for omit in (False, True):
            got = aes_encrypt(pt, key, nr=nr, omit_final_mc=omit).cpu().tolist()
            for i in range(n):
                exp = ref_encrypt(pt_l[i], k_l[i], nr, omit)
                assert got[i] == exp, (
                    f"nr={nr} omit_final_mc={omit} 样本{i} 不一致\n"
                    f"  got={got[i]}\n  exp={exp}"
                )
    print(f"[2] 逐轮比对 (nr=1..10 × 末轮含/省列混合, {n} 组随机样本)")
    print("    ✓ 全部一致")


def test_batch_consistency():
    """同一输入重复调用结果一致（确认没有原地修改输入）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randint(0, 256, (8, 16), dtype=torch.int64, device=device)
    k = torch.randint(0, 256, (8, 16), dtype=torch.int64, device=device)
    x0 = x.clone()
    a = aes_encrypt(x, k, nr=2, omit_final_mc=False)
    b = aes_encrypt(x, k, nr=2, omit_final_mc=False)
    assert torch.equal(a, b) and torch.equal(x, x0)
    print("[3] 批次一致性与输入不被原地修改: ✓ 通过")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    test_fips197()
    test_round_by_round()
    test_batch_consistency()
    print("\n全部通过 ✓")
