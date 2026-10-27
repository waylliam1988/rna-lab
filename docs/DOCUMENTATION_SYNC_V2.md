# 文档同步与下一步教学 v2（2026-10-28）

承接点c2adc36，本次核对c3f7c2f之后37个提交及既有源码/聚合报告。只同步文档，不新增模型训练、网络请求、数据审核决定或数据放行；不会把之前暂停的研究阶段标为已恢复。原实验协议、机器结果、原始数据和权重保持原版本。

| 滞后位置 | 更正 |
| --- | --- |
| 根README | 撤下“尚未开始新实验”、旧61条默认入口和小模型命令说明；同步CNN/joint/BCE入口、loss v2、数据版本与当前阻塞 |
| STATUS | 日期更新为2026-10-28；S0/S2及关系机器审计已执行，S1人工关系未完，loss v2已完成而A/B/C未启动；最近测试数是历史130项记录 |
| DATA_FIRST_EXECUTION_V1 | 共享损失位置改为losses.py；前四项清单已存在，A/B/C产物仍待生成；损失候选已完成v2、gate失败、停止合入 |
| DATA_FIRST_ROADMAP_V1、ROADMAP | 保留未来验收设计，标注已执行与未完成项；修正默认PairBaseline描述，不重做S0 |
| BASELINE_DECISION_V1 | 新增后续loss v2口径，区分旧逐RNAF1、组均匀采样新逐RNA/组F1，不宣称独立泛化改善 |
| NEXT_STEPS、DATA_FIRST_PROGRESS_V1、STAGED_VALIDATION | 增加当前指南及执行范围，修正“尚无运行”仅指A/B/C，保留逐条历史日志 |
| data/README、DISCUSSION、旧文档审计 | 指向已完成审计及当前续接，不再引导默认重复获取或将旧审计当当前状态 |
| NEXT_ACTION_GUIDE_V1（新增） | 小批次原文→构建体→同源/历史暴露→去向与标签复核→新版本→辅助mask/冒烟→冻结A/B/C；给出证据字段与停止分支 |

## 当前继续条件

新辅助监督与新独立测试准入均为0，已有88/10探索划分不因这句话被否定。先按已有文献/entry候选进行小批次人工证据审核；找到DOI不是构建体/同源独立性证明。无法核实则保留unknown与隔离原因，不新增下载量或降低门槛。配对级mask、评价分母与两阶段链路验收未完成前，不启动A/B/C。

## 本次核验范围

修改文档的本地链接目标和git差异格式检查；三个通用入口--help加载检查。只检查帮助文本，不执行训练/评价；不重跑130项历史测试，也不把文档校验算作模型验证。机器摘要见`data/manifests/documentation_sync_v2.json`，记录实际检查数、入口核验和文件指纹。

教学入口：[NEXT_ACTION_GUIDE_V1.md](NEXT_ACTION_GUIDE_V1.md)；当前台账：[DATA_FIRST_PROGRESS_V1.md](DATA_FIRST_PROGRESS_V1.md)。
