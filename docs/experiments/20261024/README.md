# 2026-10-24：两轮AES区分器与三轮16位子密钥恢复

## 文件索引

| 文件 | 含义 |
|---|---|
| config.json | 原始正式训练参数、种子、环境与源码哈希 |
| history.json / train.log | 20轮实际训练记录 |
| test_metrics.json | 云端最终独立测试，绑定最佳权重SHA256 |
| independent_classification.json | 本机新100万数据、显式规则及第14/15轮复测 |
| attack_summary.json | 1000密钥配对攻击统计及Wilson区间 |
| attack_trials.json / .csv | 3000条逐次结果，含排名、真值和猜测 |
| metadata.json | 补充测试的环境、种子和原评估脚本哈希 |
| source/ | 训练时核心源代码快照，用于核对config中的哈希 |

attack_trials内的密钥仅来自本地模拟实验，不是任何真实系统的凭据。三组共享1000个密钥，并分别使用同一实例前1/2/3对明文。当前仓库不包含云端连接信息及个人本地路径。

## 如何理解文件中的版本

原始config和metadata的哈希描述当时实际运行的版本。公开入口保留核心训练/攻击计算；评估入口增加了可选的原始完整运行目录，使没有全部20轮checkpoint的读者也能使用最佳权重复测。该发布入口不冒充当时原评估脚本的字节级副本。

config中outdir为原实验相对目录，仅作记录；复测请传新的outdir。保留原始JSON字段，不为了适应公开目录而改写历史训练参数。

## 重跑配对评估

在仓库根目录执行：

```bash
python evaluate_experiments.py --checkpoint checkpoints/net_r2b01.pt --outdir runs/review_new --trials 1000 --samples 1000000 --device cuda
```

如果另有原始完整训练目录，可再加 `--backup-run 完整目录`，检查最佳模型与epoch001的张量一致性，并加载第14、15轮做额外推理。仓库只随附最佳权重，默认跳过这两轮的诊断，但保留当时诊断结果。

图表对应这个已归档实验，`plot_log.py` 从此目录读取历史及统计，输出到docs。
