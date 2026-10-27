# RNA Lab — 从可验证的RNA 2D基线开始

主线是仅输入1D序列预测2D配对图。原Nezha项目保持独立；Ghost、RWKV和分步更新保留作逐项验证的研究机制。

## 当前进度（2026-10-28）

- 约150k参数CNN参照已冻结；训练/评价/推理入口已统一模型加载。默认训练数据为`joint_split_v2`，训练和评价核验manifest与数据指纹。
- 基线冻结、数据版本核对、隔离原因及profile/历史/比较候选/实验结构关系审计已完成相应机器审核；生物学来源关系仍有缺口。没有新增辅助监督或新独立测试获准使用。
- BCE与BCE＋伙伴竞争按v2协议完成6次训练和6个检查点独立重载。验证组宏F1均值0.5224与0.5248，未过预声明门槛；保持balanced BCE默认，停止该项调优。
- 最新有限RCSB/Crossref文献元数据核对完成，8条记录仍缺至少一个关联主研究键；候选DOI未写入关系图。上一研究阶段已收尾暂停；本次只同步文档和操作指南。
- 最近一次代码回归记录为130项测试通过；这是代码/契约证据，不是模型准确率证明。

现在先读[下一步操作指南](docs/NEXT_ACTION_GUIDE_V1.md)。唯一逐项当前台账是[DATA_FIRST_PROGRESS_V1.md](docs/DATA_FIRST_PROGRESS_V1.md)；[STATUS](docs/STATUS.md)作概览，[NEXT_STEPS](docs/NEXT_STEPS.md)作续接入口。

## 数据怎么理解

| 版本/类型 | 已有数量 | 当前用途 |
| --- | ---: | --- |
| `annotation_review_v1`扩展2D池 | 176条 | joint划分的源版本；不是176条均可训练 |
| `joint_split_v2` | train88 / validation10 / test8；另70条隔离 | 已放行探索子集；旧8条test已揭盲，不再调参或包装为新盲测 |
| `clean_release_v1`实验关联1D | 213579条不同序列 | 联合去向中213473条隔离；不启动全池1D预训练 |
| `clean_release_v1`2D | 141条不同序列 | 与joint同序列74条，其中71条标签一致、3条冲突；其余67条仍未建立独立性准入，不与176相加 |
| 实测探测profile | 344495个 / 212652条不同序列 | 反应性不是配对图；本轮未准入辅助训练 |

141条clean-release记录的来源、链和规范序列映射已复核，180个关联结构条目均有哈希回连；133条记录的所有关联条目具有主研究键，8条仍缺至少一个。具有研究键不等于构建体和家族独立性已审核通过。没有新的准入不代表此前全部实验数据无效；已有88/10探索划分保留。

比较分析只能成为独立辅助池，需原文、比对映射、标签范围和联合隔离证据；模型预测/补全标签仍排除。未知标签不当负边。详见[数据审计](docs/DATA_FIRST_AUDIT_V1.md)和[比较数据政策](docs/COMPARATIVE_DATA.md)。

## 模型与证据边界

当前默认`SequencePairModel(kind=cnn,width=113,hidden=64,depth=2)`，150019参数；早期`PairBaseline`仅作显式回退。CNN旧参照训练宏F1为0.9723–0.9902，旧比较开发逐RNA宏F1为0.4332±0.0419；新损失对照采用组均匀采样，BCE逐RNA宏F1为0.4710、组宏F1为0.5224。这些是不同实验/汇总，不能混为同口径提升或新测试成绩。

CNN是当前能充分拟合且成本低的工程参照，不宣称架构泛化更优。其他编码器有限lr扫描已止损，不重开排名。预测范围为保守AU/GC/GU单伙伴配对及观测标签范围；贪心解码允许交叉边，不等于假结或真实动力学已解决。sigmoid是未校准分数。

原物理接口和35/5静态3D/模拟辅助支线已验证部分接口与状态变化，不代表真实共转录折叠或序列到3D预测成功；J能量门控仍关闭。历史结果入口：[基线决策](docs/BASELINE_DECISION_V1.md)、[拟合扫描](docs/ENCODER_LR_SCAN_RESULTS_V1.md)、[损失v2](docs/DATA_FIRST_LOSS_RESULTS_V2.md)、[三关检查](docs/GATES_VALIDATION_V1.md)、[输入v2](docs/INPUT_REPRESENTATION_V2_RESULTS.md)。

## 本机使用（PowerShell）

```powershell
Set-Location 'D:\RNA flod'
./run.ps1 scripts/train_baseline.py --help
./run.ps1 scripts/evaluate.py --help
./run.ps1 scripts/predict.py --help
```

以上仅查看入口，不训练、不下载、不评价模型。`run.ps1`使用项目Python3.12.8虚拟环境，PyTorch2.7.1+cu128；单次实验按单卡约22GB显存安排。详细环境见[ENVIRONMENT](docs/ENVIRONMENT.md)。

训练入口默认CNN、joint_split_v2和balanced BCE，按validation组宏F1保存开发检查点。一般CLI训练不等于冻结实验复现；评价CLI只开放validation，核验数据指纹，旧检查点缺指纹须显式兼容选项。新独立test需另立冻结协议。不要重跑早期下载/prepare脚本来“刷新”数据，也不要覆盖既有运行目录。

后续A/B/C尚未开始：辅助监督准入、配对级mask、评价分母与两阶段冒烟须先通过，才冻结预算和正式运行。见[执行细则](docs/DATA_FIRST_EXECUTION_V1.md)和[路线图](docs/DATA_FIRST_ROADMAP_V1.md)。

仓库提交源码、文档和聚合摘要；原始材料、逐记录台账、环境和权重留本地。[数据目录](data/README.md) · [迁移边界](docs/MIGRATION.md) · [最新文档同步](docs/DOCUMENTATION_SYNC_V2.md)。
