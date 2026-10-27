# RNA Lab — 从可验证的 RNA 2D 基线开始

本项目从 **1D 序列 → 2D 碱基配对图** 开始。旧项目保持独立。Ghost 动力学、RWKV 记忆和分步折叠是后续主线，需要分别验证其收益。

最新：[训练拟合验收](docs/ENCODER_LR_SCAN_RESULTS_V1.md)已完成。EGNN几何审计与有限lr扫描已完成：坐标/距离进入网络，padding对齐、坐标更新与梯度通过；输入是序列位置种子，不是实验几何。按冻结协议完成3编码器×4变体×3种子、25epoch，仅评估88条训练RNA。Transformer/EGNN-seed/RWKV-Lite最终训练F1最高为0.7434/0.6438/0.7650，原CNN三个种子为0.9723–0.9902。全部低于0.8的止损条件触发，停止调优；0.9拟合验收门槛未通过，不进行新的验证集比较，不宣称CNN泛化更好或架构优劣。36检查点重载、82项测试通过；旧速度/显存仅作为当前实现的工程比较。

最新：[输入表示v2实验](docs/INPUT_REPRESENTATION_V2_RESULTS.md)已完成。正式区分静态上下文与动态观测；Ghost从指定静态密度得到小幅开发收益。实时几何接触的收益依架构/组而异，移除指定静态项的MLP分支达到预声明一致性规则；其余平均改善不足以证明普遍收益。24检查点重载、76项测试通过，J门控保持关闭。

最新：[三关任务与接口检查](docs/GATES_VALIDATION_V1.md)已完成。受检时序/未来隔离/动作接口和小样本拟合通过；新模拟任务能学习部分状态与动作变化，但未建立RWKV额外记忆的稳定收益。7种对照×3种子、21个检查点重载通过，72项测试通过；仅5组开发验证，不是实验折叠轨迹或独立盲测。

逐项完成、验证与提交的最新进度：[STAGED_VALIDATION.md](docs/STAGED_VALIDATION.md)。扩展配对池和隔离划分使用独立版本，原始清洗发布不覆盖。

六项初步验证已完成，见[固定协议实验结果](docs/EXPERIMENT_RESULTS_V1.md)：176条配对标签中106条进入保守隔离划分，88/10/8；三种子比较与封存测试均未支持复杂机制稳定增益。RWKV状态预测未胜过持久性基线，不称已验证的世界模型。

上述结果针对新写的2D适配器，不能替代原Nezha完整体系的验证。[迁移边界核查](docs/NEZHA_FIDELITY_AND_3D_PLAN.md)后，已恢复[原物理状态、观测和反馈接口](docs/PHYSICAL_INTERFACE_V1.md)，并完成[静态3D联合隔离](docs/THREE_D_ISOLATION_V1.md)。新物理预测实验按[预先冻结协议](docs/PHYSICAL_FORECAST_PROTOCOL_V1.md)分别检查训练预算和展开长度；模拟转移与实验坐标明确区分。

**配对监督只用通过审核的直接实验证据。** 扩充获取、官方来源复核及精细清洗见[本轮发布报告](docs/CLEAN_RELEASE_V1_REPORT.md)和[精细清洗协议](docs/FINE_CLEANING_PROTOCOL_V2.md)。实测反应性与精确配对标签分开计数，未知/未核验记录隔离，不用模型预测或补全值充数。

新清洗池位于 `data/experimental/clean_release_v1`，全池不自动用于训练。后续扩展2D池176条已有 `joint_split_v2` 的88/10/8隔离子集；70条2D和213473条1D继续隔离，1D预训练关闭。旧8条测试已揭盲，不再用于调参。原61条、49/6/6划分的[链路验证版本](docs/DATASET_V0_2_REPORT.md)保留，默认训练路径不变；此前旧基准烟测不代表严格实验数据下的效果。

可靠的比较序列分析可以作为**独立辅助监督池**，经过逐条溯源、比对映射与同源性审核后，再用实验留出集验证收益；不把它改名为实验真值。详见[比较分析数据策略](docs/COMPARATIVE_DATA.md)。模型预测结构和模型补全值仍排除。

仓库只提交源码、文档及数据策略；第三方数据、环境和权重保留在本地。数据获取方式见[data目录说明](data/README.md)。

## 已有内容

- 官方基准归档下载、SHA256 清单、严格 BPSEQ 解析、清洗和审计报告。
- 保留官方 train/validation/test 划分；逐序列去重，移除跨集合完全相同序列。
- 小型 CNN 配对基线、独立训练/评估、允许交叉配对的贪心解码和指标。
- 研究路线、讨论纠错、数据说明及旧 Ghost/RWKV 的只读参考快照。
- `src/rna2d/core` 保留独立 RWKV、连续控制层、显式状态 Ghost；已通过 `iterative.py` / `ghost_frontend.py` 接入实验分步2D预测器，静态基线独立保留，结果按消融报告。

## 快速开始（PowerShell，在本目录运行）

正式环境使用 Python **3.12.8**，解释器位于 `.venv3128/Scripts/python.exe`。实验审计还使用 requests、Gemmi、NumPy、SciPy、MDTraj、Barnaba 等已锁定依赖。统一通过 `run.ps1` 调用，避免误用 PATH 或临时 CPU 环境。

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
# 此入口仅为早期小模型示例，不是约150k比较版复现命令
# 旧测试已揭盲，不再用于后续方案验收；新盲测需另行冻结
```

`run.ps1` 固定调用项目的 Python 3.12.8 环境；不修改全局 PATH。选择 GPU 可用 `--device cuda:0` / `cuda:1` / `cuda:2`。初版训练是单卡，不自动合并三张卡的显存。

## 必须理解的边界

扩充全池完成的是**精确序列去重和证据/质量审核**；仅joint_split_v2中已放行子集完成声明范围的联合同源/研究/构建体隔离，不代表全量1D均可训练。原61条版本只在已声明的RNA3DB分量与80%全局相似性范围内做了隔离。历史 `data/processed` 基准来源混合，不属于新实验池。长度、配对类型、背景与掩码范围会改变评估分布，不能直接与论文全量成绩比较。

基线只预测 AU/GC/GU、一碱基至多一配对的任务；含其他配对的样本完整保存在标准化记录中，但从基线集合排除，不静默改成未配对。贪心解码支持交叉边，但不保证最优，也不代表已经解决假结预测。未经校准的 sigmoid 输出称为分数。

当前CNN版本选择及限制见[基线决策](docs/BASELINE_DECISION_V1.md)。约150k比较版与默认train_baseline.py的早期小模型不同，不混用指标。

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
