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

本节为首轮报告形成时的状态；后续profile/历史、比较候选和clean_release候选关系审计已补入下文。当前`joint_split_v2`只支持冻结探索，不能支持广泛跨家族泛化；A/B/C因辅助监督无准入数据继续不启动。伙伴竞争损失对照已按v2协议完成，预声明gate不通过，候选不合入、默认仍为BCE，详见[阶段6D结果](DATA_FIRST_LOSS_RESULTS_V2.md)。生产训练/评价/推理入口现默认使用joint manifest约束的150k CNN；PairBaseline保留显式回退，smoke不构成新效果证据。后续从S1-1未完成的生物学来源关系审核继续；数据来源、构建体、同源或研究独立性无法核实时继续隔离。

产物：`data/manifests/data_first_baseline_v1.json`、`data/manifests/data_first_relations_v1.jsonl`、`data/manifests/data_first_admission_v1.jsonl`、`docs/DATA_FIRST_AUDIT_V1.json`。

## 后续精确关系审计（2026-10-28）

S1后续审计分别见[profile与历史关系报告](../data/manifests/data_first_profile_history_audit_v1.json)及[比较候选关系报告](../data/manifests/data_first_comparative_candidate_relations_v1.json)。344,495个profile全部落在quarantine，仍无配对标签；167个RMDB来源文件字节哈希均与本地收据一致。历史逐记录明细留在本地并由Git忽略。

本地bpRNA-PDB候选669条、477条唯一序列全部可从本地压缩包重建，但URL/下载时间收据、原始PDB/链、方法/覆盖和提取映射未核验；所有来源/质量/准入标记均为false。与当前joint train/validation精确序列重合的候选边集都不同于当前pair labels，这只是未审核BPSEQ与当前标签的关系，不构成标签裁决。候选训练和新测试准入均为0。

clean_release的141条已有直接坐标来源、质量与掩码标签；74条与joint精确同序列，其中71条mask/pairs相同，3条有冲突。其余67条按RNA3DB非占位组件、主要文献、PDB条目和冻结分割使用的局部相似性规则与joint关联后，仍有当前split连接、历史test暴露和缺母本/构建体/同源信息的组；现有审计没有准入新测试。聚合结果见[clean_release评估候选报告](../data/manifests/data_first_clean_release_eval_candidate_audit_v1.json)。

后续只读解析本地CIF的聚合结果见[clean_release CIF元数据审计](../data/manifests/data_first_clean_release_cif_metadata_audit_v1.json)。141/141条记录的来源下载收据SHA256和解压CIF SHA256均匹配；180个关联结构条目全部可从本地收据回连并通过文件哈希。141/141条选中链、实体与规范聚合序列互相匹配。CIF均含主引用行和题名，135/141条有DOI或PubMed主研究键；将全部替代结构条目纳入后，170/180个条目有主研究键，133/141条clean-release记录的全部关联条目均有主研究键，另8条仍缺至少一个。141条记录都没有显式构建体祖先字段；实体来源方法与描述字段齐全，少量记录还注明突变/片段，但这些信息不能证明母本构建体或实验独立。

对上述10个缺主研究键的关联结构条目又做了限定的官方RCSB Data API entry metadata复核，见[官方citation复核报告](../data/manifests/data_first_clean_release_remote_citations_v2.json)。10/10响应身份校验通过，主引文行和题名均存在，但没有可用主DOI或PubMed键；另见10/10有PDB entry DOI字段，该字段属于结构条目，不计作论文引用。8条记录的研究关系缺口没有被消除。初版汇总只解析了便捷主引文对象，随后从已保存的原始响应重解析citation原始表并生成v2；未重复请求。官方数据库元数据只作来源交叉核验，不能替代原论文、构建体祖先、家族/同源或评估独立性审查。

随后只对这10个已核实的主引文题名各做一次Crossref works题名查询，逐条响应本地保存并校验哈希。聚合复核见[Crossref题名审计](../data/manifests/data_first_clean_release_crossref_review_v2.json)：5个entry有精确题名候选（共6项），4个entry有首作者相符且年份未冲突的候选（共5项），但这5项的来源年份均未知，因此严格题名/年份/首作者匹配为0。5个候选DOI未加入关系图；133/141条记录的主研究键覆盖未变，8条仍缺至少一个。初次离线重解析因未定义的校验函数失败，修正为直接核对响应entry身份后成功；没有重发网络请求。Crossref记录仅供后续人工核对，不能证明候选DOI确属主论文或建立构建体/家族独立性；训练与新测试准入仍为0。

这些审计没有读模型预测或旧测试表现，没有训练，也没有更改标签/划分。新增CIF元数据补足部分来源和链/实体关系，但不能证明远程同源、构建体祖先、实验独立或评估盲法；新训练/测试准入仍为0。旧test已揭盲，继续维持无新盲测结论。逐记录关系表保留本地，不推送第三方记录ID和序列哈希。
