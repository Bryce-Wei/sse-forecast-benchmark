# 行情缓存

在仓库根目录运行 `python scripts/fetch_data.py` 下载上证指数 `000001.SS` 的固定历史区间：2023-03-01 至 2026-09-11。

原始行情 JSON 与下载日志只保存在本机，已加入 `.gitignore`。也可以使用 `python scripts/fetch_data.py --from-file /path/to/yahoo_raw.json` 导入已有 Yahoo chart 响应；支持通过 `SSE_DATA_PATH` 环境变量指定本地缓存路径。

程序核验指数代码、日期唯一性与有效收盘价，不会填补缺失行情。数据源可能修订历史数据或暂时限流；新下载的数据不保证与原实验的字节完全相同。每次运行记录日期与收盘价的规范化 SHA-256，结果写入 `runs/`；`results/reference/` 的已发表结果不受影响。

行情的使用和再分发仍遵循数据提供方的条款；代码许可证不授予第三方行情数据的权利。
