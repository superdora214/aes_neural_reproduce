"""
Bit-Sliced 卷积 ResNet（神经区分器网络）
=========================================
**本文件从 `gohr_pytorch_reproduce/src/model/distinguisher.py` 原样沿用网络结构**，
只把默认的 `word_size` 从 16 改成 8，以适配 AES 的字节级输入。

张柳2024 对网络的处理也是"原样照搬 Gohr 架构，只为适配 AES 密文的数据格式做调整"，
所以这个沿用是有依据的，不是我们自己发明的改造。

形状映射（Gohr → 本文 AES）：
  输入 [N, 2*nbytes*8]
  → view [N, channels_in, word_size]
      Gohr : channels_in = 4 = 2 个密文 × 每个 2 个 16 位字，word_size = 16
      AES  : channels_in = 4 = 2 个密文 × 每个 2 个字节，  word_size = 8
  → 1×1 卷积扩到 32 通道（跨通道融合，这里就对应"两个密文/两个字节之间"的信息）
  → depth 个残差块（核宽 ks=3，逐位相邻）
  → 预测头 Flatten → 64 → 64 → 1 → Sigmoid

⚠️ 与 Gohr 版一致的注意事项：view 必须是 [N, 通道, 字长]，
   不能写成 view(N, 字长, 通道) 再 permute —— 那会破坏位的分组结构。
"""
import torch
import torch.nn as nn


class BitSlicedResNet(nn.Module):
    """
    注意：张柳2024 第 2.2 节明确写了 "an initial convolution layer with width-1 and
    **Nf = 16** channels"，且卷积块也用同一个 Nf，所以本实现把默认通道数设为 16
    （Gohr 在 Speck32/64 上用的是 32）。残差塔**保留**——原文："A skip connection from
    the output of the block's final ReLU layer to its input ... five such blocks"。
    """

    def __init__(self, num_blocks=2, num_filters=16, word_size=8,
                 d1=64, d2=64, ks=3, depth=5):
        super().__init__()
        self.word_size = word_size
        self.channels_in = 2 * num_blocks

        # ① 初始 1×1 卷积
        self.conv0 = nn.Conv1d(self.channels_in, num_filters, kernel_size=1)
        self.bn0 = nn.BatchNorm1d(num_filters)

        # ② 残差塔
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(num_filters, num_filters, ks, padding=ks // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, ks, padding=ks // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            ))

        # ③ 预测头
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * word_size, d1),
            nn.BatchNorm1d(d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.BatchNorm1d(d2),
            nn.ReLU(),
            nn.Linear(d2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        n = x.shape[0]
        x = x.view(n, self.channels_in, self.word_size).float()
        x = torch.relu(self.bn0(self.conv0(x)))
        shortcut = x
        for blk in self.blocks:
            x = blk(x) + shortcut
            shortcut = x
        return self.head(x)


def make_distinguisher(depth=5, **kwargs):
    return BitSlicedResNet(depth=depth, **kwargs)
