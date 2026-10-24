# 最佳权重

`net_r2b01.pt` 是完整规模训练按验证纯 MSE 最低选出的第 1 轮模型，不是早期小测试权重。

- 两轮 AES；末轮包含 MixColumns；字节 `{0,1}`。
- 当前 PyTorch 输入 `[N,4,8]`，5 个残差块，16 个卷积通道。
- SHA256：`7e1bd6d6fb073c19691e187e865feb60ba3d13409cf4af34d0ebfedf4415a4f5`。
- 该文件是模型 state_dict，没有优化器或完整续训状态。

用 `key_recovery.py --checkpoint checkpoints/net_r2b01.pt` 加载；无 CUDA 时加 `--device cpu`。
