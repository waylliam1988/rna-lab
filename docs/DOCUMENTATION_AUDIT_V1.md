# 文档同步审计（2026-10-27）

检查范围：仓库根README、AGENTS、data/README和docs中的Markdown；对照当前源码、已有实验摘要和冻结协议。只更正状态与解释，不改变历史指标、数据、权重或冻结实验协议。各阶段的13/50/80项测试数是当时记录，不一律替换成当前82。

| 发现的滞后 | 更新位置 | 处理 |
| --- | --- | --- |
| 路线仍停留下载/短训练，3D一直写作未来 | [ROADMAP](ROADMAP.md)、[DISCUSSION](DISCUSSION.md) | 列明已完成阶段；当前固定CNN工程参照，停止本轮调优 |
| 全池与放行子集混淆 | 根README、data/README、[清洗发布](CLEAN_RELEASE_V1_REPORT.md)、[历史数据协议](DATA_PROTOCOL.md)、[旧61条版本](DATASET_V0_2_REPORT.md) | 区分全池、176条扩展2D、88/10/8子集、隔离1D和已揭盲测试 |
| 初版未迁入描述遗漏后续物理恢复 | [MIGRATION](MIGRATION.md)、[迁移预审](NEZHA_FIDELITY_AND_3D_PLAN.md)、[物理接口](PHYSICAL_INTERFACE_V1.md) | 更新恢复范围，保留完整BioSystem/旧权重尚未恢复的边界 |
| 通道审计仍说静态消融和实时接触待做 | [通道审计](CONTACT_CHANNEL_AUDIT_V1.md) | 链接已完成输入v2，保留原时点语义 |
| VC++警告及初版未训练限制看似仍生效 | [ENVIRONMENT](ENVIRONMENT.md)、[STATUS](STATUS.md) | 明确历史状态，系统运行库已安装；阶段计数不篡改 |
| 旧比较下一步仍暗示继续调参或按验证排序 | [比较结论](ENCODER_COMPARISON_FINDINGS_V1.md)、[比较结果](ENCODER_COMPARISON_RESULTS_V1.md)及报告生成器文字 | 明确扫描已止损，旧验证数值不改，不作架构泛化归因 |
| “CNN”混淆早期小模型与比较版 | [基线决策](BASELINE_DECISION_V1.md)、根README、[下一步](NEXT_STEPS.md)、[进度](STAGED_VALIDATION.md) | 记录约150k比较版的类/宽度与证据，保留早期入口，不自动替换训练器 |
| 早期实验“暂不升级3D”被当作当前计划 | [早期实验](EXPERIMENT_RESULTS_V1.md) | 增加历史范围说明和当前入口，原数值与测试结论保留 |

基线决策：当前选择CNN是因为可充分拟合、运行成本低、适合固定受控对照；不意味着高精度或比其他架构泛化更好。新数据和独立冻结评估优先于继续追逐编码器排名，J门控保持关闭。

核验：修改的Markdown本地链接目标检查、git diff --check、报告生成器语法检查；未启动训练、读取测试预测或重复下载数据。检查摘要见data/manifests/documentation_audit_v1.json。
