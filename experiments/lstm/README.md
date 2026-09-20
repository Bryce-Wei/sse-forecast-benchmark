# 独立 LSTM 滚动预测实验

本目录是唯一的 LSTM 训练实现。它可以独立运行，也由 PatchTST 和 TimesFM
对照实验直接调用。两组比较不会再分别维护或训练自己的 LSTM 副本。

在仓库根目录安装依赖并准备历史数据后执行：

```bash
# 各验证第一个预测日期，仍使用完整 40 轮选轮与重新训练
python experiments/lstm/run_all.py --history 20 --limit 1
python experiments/lstm/run_all.py --history 32 --limit 1

# 完整 64 日实验及图表
python experiments/lstm/run_all.py --history 20
python experiments/lstm/run_all.py --history 32 --language en
```

输出分别保存在 `runs/lstm/history_20/` 和 `runs/lstm/history_32/`。每个目录包含
协议、逐日预测与训练审计、保存模型及重载检查。完成 64 个日期后才生成
`metrics.json`、`predictions.json`、`observations.json` 和预测图。`--limit` 的
局部验证不会被标为完整实验。`--smoke` 只检查前向/反向；`--reload-only`
检查已经保存的模型。

测试日期为 2026-06-15 至 2026-09-11。每次预测使用此前两年日历区间内的
收盘数据训练，输入最近连续 20 或 32 个交易观测，预测下一交易观测的收盘。
内部最后 40 个历史目标日期选择 1–40 轮中的最佳轮次；随后丢弃模型和
优化器，以相同初始 seed 从头在全部历史数据上训练选定轮数。验证标准化
排除验证观测，最终重训标准化只使用预测日前的历史。

结构保持 `LSTM(32, return_sequences=True) -> LSTM(16, relu) -> Dense(1)`，
7,505 个参数。MAE 损失、Adam 0.001、batch 32、梯度 clip value 1.0、seed 88
均保持原协议。此次重构没有重新调参。模型不是 LSTM 架构的原创实现；
使用 TensorFlow/Keras 层，实验结构延续课堂示例。

`protocol.load_records(history, ...)` 是配对实验读取结果的公共接口。它检查
数据指纹、输入长度、模型配置、训练与验证日期、标准化参数、选定轮数、
优化器步数和保存模型位置。20/32 日输出分开，禁止交叉读取缓存；更换
数据或协议时也不会悄悄复用旧结果。旧的 `runs/patchtst/per_day/lstm/` 和
`runs/timesfm/per_day/lstm/` 不再被读取。

预测图左侧为真实历史，右侧为逐日单步预测与测试真实值，并提供测试区间
放大图。整条三个月曲线由每日预测连接而成，不是在起点一次预测三个月。
这些日期此前已经查看过，属于历史复核，不能称为全新独立测试。

无需安装 TensorFlow 即可执行数据协议的单元检查：

```bash
python -m unittest experiments.lstm.test_protocol
```
