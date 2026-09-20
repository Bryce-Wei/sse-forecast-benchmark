# ARIMA：收益率差分、少量参数和训练窗口比较

这里的脚本由本项目根据课堂 ARIMA 实验结构重新实现，没有复制课堂原文件。模型由 `statsmodels.tsa.arima.model.ARIMA` 提供；项目贡献是固定日期、滚动预测、验证期选择、基准评估、拟合记录和可复现图表。

## 方法

- 行情：根目录下载脚本提供的 `data/yahoo_sse.json`，上证指数 `000001.SS` 日收盘。所有结果截断于 **2026-09-11**。
- 目标：日对数收益 `log(close_t / close_(t-1))`。拟合时乘 100，预测后还原；预测点位为 `previous_close * exp(predicted_log_return)`，不作均值偏差修正。
- 原模型：**ARIMA(5,1,0)**，从 2023-03-14 的收益开始，逐日扩展历史并重拟合。先保存预测，再加入目标日的真实收益；不是在某一天一次性预测半年。
- 验证：**2025-03-14 至 2026-03-13，共 242 个目标日**。
- 历史复核：**2026-03-16 至 2026-09-11，共 125 个目标日**。这一时期在前期探索中已看过，不能当作全新未接触的测试集。
- 比较六个模型配置与三个训练窗口，共 **18 个候选**。窗口为最近 126、252 条收益或从 2023-03-14 起扩展。

| 配置 | trend | 含义 |
|---|---|---|
| (5,1,0) | n | 原模型，在收益上再作一次差分 |
| (5,0,0) | n | 同阶 AR，取消收益上的额外差分 |
| (0,0,0) | c | 历史均值 |
| (1,0,0) | c | AR(1) |
| (0,0,1) | c | MA(1) |
| (1,0,1) | c | ARMA(1,1) |

主候选按验证期合并后的**点位 RMSE**选择，数值相同时按配置 ID 排序；另外冻结每个训练窗口的验证赢家作为辅助对照。历史复核结果不用于重新选模型。若候选在验证期未超过前日收盘基准，协议保留基准作为默认方案，同时仍报告候选结果。

ADF、KPSS 和一阶 ACF 分别检查对数点位、收益率、收益再差分；检验只用对应截止日之前的数据。拟合不收敛时按 `lbfgs 200 → lbfgs 1000 → powell 1000` 固定重试，最终失败使整个配置不合格，不删除困难日期。稳定性分析包含两个时段、各日历月和 5/10 日循环区块 bootstrap，每种区块长度 5,000 次。bootstrap 未校正模型搜索。

## 运行

在仓库根目录完成根 README 中的数据准备与基础依赖安装后：

```bash
# 小规模真实拟合：原模型前 2 天，18 个候选各拟合 1 个验证日
python experiments/arima/run_all.py --smoke

# 完整原版 → 18 候选验证与历史复核 → 图表
python experiments/arima/run_all.py --workers 3
```

ARIMA 所需依赖为 `numpy`、`pandas`、`statsmodels`、`matplotlib`、`threadpoolctl`，不需要 TensorFlow、PyTorch 或下载模型权重。完整运行包含数千次拟合，耗时取决于 CPU。减少 `--workers` 可降低并行资源占用。

也可分步运行：

```bash
python experiments/arima/run_forecast.py
python experiments/arima/compare_models.py --workers 3
python experiments/arima/render_comparison.py
# 已有完整原版预测时，只重绘原版图
python experiments/arima/run_forecast.py --plot-only
```

所有路径从脚本所在仓库定位，不依赖当前工作目录或 Windows 用户目录。图表默认使用 Matplotlib 自带 DejaVu Sans 和英文标签；绘图入口和 `run_all.py` 可加 `--language zh` 使用已安装的中文字体。

原版与改进版的点位图、收益率图均采用两层布局：上图左侧为首次预测前的真实历史，右侧为预测与真实值对比，下图放大测试区。点位图通过根目录 `forecast_plotting.py` 与其余模型共用布局。重画 README 及原版 PDF 图请使用根目录的 `scripts/render_report_figures.py`。

## 输出

- `runs/arima_original/`：原模型的预测明细、指标、收益图和点位图。
- `runs/arima_improved/`：差分诊断、冻结协议、18 候选验证排名、冻结选择、历史复核预测、稳定性指标、3 张图和 `comparison_report.md`。
- `runs/arima_improved/cache/`：逐候选拟合缓存；使用前核验规范化日期/收盘数据 SHA-256、模型配置和阶段。
- `runs/arima_smoke/`：独立的冒烟结果，不参与正式模型选择，不覆盖完整结果，不代表复现了完整性能指标。

原版和改进版必须使用相同的行情快照，程序会核验规范化日期/收盘记录的 SHA-256（不包含供应商可变元数据）。已有改进协议若与当前数据不一致，程序会停止，避免混用两次实验；请先将旧 `runs/arima_original/` 和 `runs/arima_improved/` 移至其他位置，再用新数据完整运行。数据供应商重下载的历史值可能修订，因此新下载数据不保证逐位重现报告数值。

## 已保存实验的结论

报告所用快照在验证期选中 `(0,0,0)`、含常数、最近 **252 条收益**，实质上为历史平均收益预测。历史复核点位 RMSE：原模型 **45.20**，选中候选 **41.68**，前日收盘基准 **41.42**。改进后更接近简单基准，但没有显示优于基准的稳定预测能力；不能仅凭点位曲线贴近真实价格判断模型有效。

这些是已保存报告的结果，不是硬编码的当前重跑结果；脚本会根据实际本地输入重新选择、计算并绘图。

## 参考

- [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [ADF 检验](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)
- [KPSS 检验](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html)
- [Yahoo Finance 上证指数](https://finance.yahoo.com/quote/000001.SS/history/)
