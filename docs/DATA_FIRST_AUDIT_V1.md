# 数据优先首轮审计 v1

日期：2026-10-27。此报告记录 S0 基线冻结和 S1 联合关系表的实际产物；不代表辅助池或新训练已经放行。

## 已核验

- 约150k CNN参照为 `SequencePairModel(kind=cnn,width=113,hidden=64,depth=2)`，150,019参数，float32 eager，AdamW，25 epoch，batch 4，550 次更新，种子42/43/44。
- 源码、协议、训练/验证/历史测试文件及三个 CNN 检查点均记录 SHA256。三个检查点来自既有 encoder comparison 运行，报告标记独立重载通过且未读取 test。
- `joint_split_v2` 的实际文件哈希与冻结协议一致：train 88、validation 10、test 8，组数 8/7/4；联合1D分配为 train 88、validation 10、test 8、quarantine 213473。
- `joint_split_v2_validation.json` 的检查覆盖证据、掩码/伙伴合法性、component/study/construct/sequence 隔离、局部比对边、1D 排除和历史分配保持。

## 去向决定

`joint_split_v2` 实际由 `annotation_review_v1/paired.jsonl`（176条）生成；当前审计按序列SHA256回连，并逐条比较 observed mask 与 pair labels，106条split记录全部精确匹配。clean_release_v1 的141条是另一数据版本：只按序列SHA256作别名候选，74条与split样本同序列，其中71条mask和pair labels完全一致、3条标签不同；其余67条没有split对应。clean_release别名全都不具训练资格，标签冲突记录继续隔离。训练入口仍只消费 `joint_split_v2` 文件，避免重复行或跨版本标签覆盖。

本机生成的 `data/manifests/data_first_relations_v1.jsonl` 登记这些2D记录和全部1D分配；`data/manifests/data_first_admission_v1.jsonl` 给出逐记录决定。这两份逐样本明细含第三方记录ID及序列哈希，仅留本地并由 `.gitignore` 排除。仓库只保留汇总、SHA256、生成脚本和审计规则，不推送逐样本材料。

`clean_release_v1` 的 `training_split_ready=false`，不能因为标签来自实验坐标就直接进入训练。`paired_v0_2` 只保留历史链路验证用途。对本地 `bprna_pdb_candidates_NOT_GOLD.jsonl` 固定抽取前20条做了可行性试点；20/20均缺少已验证来源和质量门禁，原始PDB/链、方法/分辨率/覆盖及提取策略仍待核对，因此结论是停止扩大并继续隔离，不是辅助训练准入。详情见 `data/manifests/data_first_comparative_pilot_v1.json`。

## 当前结论

S0 已完成，S1机器关系首版已生成并完成跨版本回连更正；S2-1的70条直接实验隔离记录均保留隔离，S2-2比较来源试点不可准入。工程 CNN 参照保留为可选基线；joint_split_v2 只支持冻结探索，不能支持广泛跨家族泛化。A/B/C 尚未冻结或运行，伙伴竞争损失只完成通用损失函数和契约测试，尚无模型效果证据。下一步补齐profile/历史关系以及独立评估来源；在A/B/C资格建立前，不开始正式辅助数据实验。

产物：`data/manifests/data_first_baseline_v1.json`、`data/manifests/data_first_relations_v1.jsonl`、`data/manifests/data_first_admission_v1.jsonl`、`docs/DATA_FIRST_AUDIT_V1.json`。
