# RNA Lab — 从可验证的 RNA 2D 基线开始

本项目从 **1D 序列 → 2D 碱基配对图** 开始。旧项目保持独立。Ghost 动力学、RWKV 记忆和分步折叠是后续主线，需要分别验证其收益。

**金标准池只用直接实验证据。** 已完成全量本地坐标清洗和保守配对数据版本：[本轮结果](docs/DATASET_V0_2_REPORT.md)。当前版本为61条序列、49/6/6划分，只适合链路和小范围验证，尚非正式泛化基准。训练入口继续拒绝未审核标签。此前旧基准烟测不代表严格实验数据下的效果。

可靠的比较序列分析可以作为**独立辅助监督池**，经过逐条溯源、比对映射与同源性审核后，再用实验留出集验证收益；不把它改名为实验真值。详见[比较分析数据策略](docs/COMPARATIVE_DATA.md)。模型预测结构和模型补全值仍排除。

仓库只提交源码、文档及数据策略；第三方数据、环境和权重保留在本地。数据获取方式见[data目录说明](data/README.md)。

## 已有内容

- 官方基准归档下载、SHA256 清单、严格 BPSEQ 解析、清洗和审计报告。
- 保留官方 train/validation/test 划分；逐序列去重，移除跨集合完全相同序列。
- 小型 CNN 配对基线、独立训练/评估、允许交叉配对的贪心解码和指标。
- 研究路线、讨论纠错、数据说明及旧 Ghost/RWKV 的只读参考快照。
- `src/rna2d/core` 已迁入独立 RWKV、连续控制层、显式状态 Ghost 核心；新旧多步数值一致性已建立测试，尚未接入 2D 基线。

## 快速开始（PowerShell，在本目录运行）

正式环境使用 Python **3.12.8**，解释器位于 `.venv3128/Scripts/python.exe`。数据脚本及数据单元测试仅需标准库。后续统一通过 `run.ps1` 调用，避免误用 PATH 或临时 CPU 环境。

```powershell
./run.ps1 scripts/download_data.py
./run.ps1 scripts/prepare_data.py --max-length 256
./run.ps1 -m unittest discover -s tests -v
```

训练环境配置见 [环境说明](docs/ENVIRONMENT.md)。当前固定 PyTorch 2.7.1 + CUDA 12.8：

```powershell
./run.ps1 scripts/check_environment.py
./run.ps1 scripts/train_baseline.py --epochs 5 --batch-size 4 --device cuda:0
./run.ps1 scripts/evaluate.py --checkpoint runs/baseline/best.pt --split validation
# 仅在方案冻结后查看测试集；测试结果不得用于调参
./run.ps1 scripts/evaluate.py --checkpoint runs/baseline/best.pt --split test
```

`run.ps1` 固定调用项目的 Python 3.12.8 环境；不修改全局 PATH。选择 GPU 可用 `--device cuda:0` / `cuda:1` / `cuda:2`。初版训练是单卡，不自动合并三张卡的显存。

## 必须理解的边界

当前数据清洗只做**精确序列隔离**，没有完成序列相似性/结构同源性聚类。基准标签不全是直接实验解析；数据不能称为纯实验金标准。长度过滤会改变基准分布，结果必须标注 `L<=256`，不得直接与论文全量成绩比较。

基线只预测 AU/GC/GU、一碱基至多一配对的任务；含其他配对的样本完整保存在标准化记录中，但从基线集合排除，不静默改成未配对。贪心解码支持交叉边，但不保证最优，也不代表已经解决假结预测。未经校准的 sigmoid 输出称为分数。

阅读顺序：[讨论与纠错](docs/DISCUSSION.md) → [研究路线](docs/ROADMAP.md) → [数据协议](docs/DATA_PROTOCOL.md) → `data/reports/prepare_report.json`。

迁移范围与验证：[核心迁移说明](docs/MIGRATION.md)。

当前可运行的数据审计：

```powershell
./run.ps1 scripts/audit_experimental_sources.py --offline
./run.ps1 scripts/screen_probing_samples.py
./run.ps1 scripts/query_current_experimental_pdb.py
```

默认数据目录为 `data/experimental/paired_v0_2`，每条记录通过实验证据门禁后才能运行。当前样本规模不支持广泛架构结论，建议先扩充数据。已有历史基准权重如需复核，可在评估命令明确指定 `--data-dir data/processed`，不能将其成绩称为严格实验证据评估。

下一次继续：[下一步清单](docs/NEXT_STEPS.md)。运行状态见 `docs/STATUS.md`（完成实测后更新）。
