# 第三方模型、代码与数据来源

本仓库是上证指数预测的复现与比较实验。PatchTST 架构、TimesFM 架构和预训练权重归各自原作者所有。本项目新增工作包括行情处理、滚动验证与训练、模型调用、基准评估、结果审计和可视化。

## PatchTST

- 官方项目：[yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)
- 论文：Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam, [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730), ICLR 2023。
- 固定源码版本：`204c21efe0b39603ad6e2ca640ef5896646ab1a9`。
- 许可：Apache License 2.0，全文见 [third_party/PatchTST/LICENSE](third_party/PatchTST/LICENSE)。
- 使用方式：`third_party/PatchTST/` 保留本实验需要的官方监督学习源码，模型源码保持原样；本项目的训练与评估逻辑位于 `experiments/patchtst/`。
- 该本地源码子集中没有上游 `NOTICE` 文件。已有源文件中的版权、归属和引用注释均保留。

## TimesFM 2.5

- 官方项目：[google-research/timesfm](https://github.com/google-research/timesfm)。
- 原作者：Google Research。
- 基础论文：[A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688), ICML 2024。2.5 的具体使用方式与权重以官方模型卡为准。
- Python 依赖：`timesfm==2.0.1`；原实验核验的源码提交为 `e56854bc9e16427c65656ee735ab4ddaa18bea67`。
- 官方权重：[google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch)。
- 固定权重版本：`1d952420fba87f3c6dee4f240de0f1a0fbc790e3`。
- 权重 SHA-256：`2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8`。
- 代码许可：[Apache License 2.0](https://github.com/google-research/timesfm/blob/e56854bc9e16427c65656ee735ab4ddaa18bea67/LICENSE)。权重许可亦为 [Apache 2.0](https://huggingface.co/google/timesfm-2.5-200m-pytorch/blob/1d952420fba87f3c6dee4f240de0f1a0fbc790e3/README.md)。
- 使用方式：通过官方 Python 包调用，权重在运行前从官方来源下载并校验；不把包的安装目录或预训练权重复制到本仓库。

## ARIMA、LSTM 与课堂示例

ARIMA 通过 statsmodels 实现；LSTM 通过 TensorFlow/Keras 实现。LSTM 的 `32 → 16（ReLU）→ Dense` 结构和最初的 ARIMA 任务受到 FINS5545 课堂示例启发。本仓库保存为本项目编写的实验、数据切分与评估脚本，不分发原始课堂材料，也不声称这些算法或网络结构为本项目首创。

其余安装依赖（NumPy、pandas、SciPy、Matplotlib、statsmodels、TensorFlow、PyTorch、Hugging Face Hub 等）继续遵循各自随包发布的许可证。

## 行情与实验结果

行情来源为 Yahoo Finance 的上证指数 `000001.SS` 日线。原始下载响应保留在使用者本机，不随源码发布。`results/reference/` 保存用于核验已发表结果的逐日预测与评分记录；它们是历史实验记录，不是数据源提供的授权数据集或未来投资信号。

根目录的 Apache 2.0 许可证适用于本仓库可授权的新增代码，不改变任何第三方代码、权重、课堂材料或行情数据的归属及条款。引用论文与许可证履行是两个不同事项。
