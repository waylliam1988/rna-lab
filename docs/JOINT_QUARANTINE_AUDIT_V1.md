# joint_split_v2 隔离池审计 v1

本阶段只审计既有隔离记录，不修改原始记录、不重建划分、不将记录移入训练或留出集。

`joint_split_v2/quarantine.jsonl` 共70条、70个序列和70个PDB条目，全部使用 `unverified_homology_or_primary_study` 原因码。64条没有已核验的 homology component；9条 primary-study 完整性不满足；6条没有 study ID。61条虽然已有 primary-study 信息，但仍缺少 homology 锚点，不能据此宣称独立家族。上下文为8条 `single_RNA`、62条 `complex`。

实验坐标证据本身没有被否定：记录均为实验坐标注释，来源和质量字段已通过现有门禁，且没有模型预测标签。当前缺口是同源、母本/构建体和研究独立性证据。已有本地材料没有足够的新证据自动补齐这些关系，因此决定为 `retain_quarantine / not_admitted`。

机器报告：[joint_quarantine_audit_v1.json](../data/manifests/joint_quarantine_audit_v1.json)。下一版本若要复核，必须新增可追溯的家族/构建体/primary-study证据并重新建立联合隔离；不能在当前冻结版本上手工改状态。
