# 数据优先工作包：当前进度与续接记录

更新日期：2026-10-27。范围：总路线图阶段0–6。这里区分旧实验、本次文档设计和未来执行，不设置人员字段。

## 当前停在哪里

**S0基线冻结已完成；S2-1直接实验隔离原因审计完成；S2-2比较来源试点结论为不可准入。** 当前从S1未覆盖的profile/历史关系与新增独立评估候选继续；不启动A/B/C，直到辅助监督具备逐条来源、映射和隔离证据并通过链路门禁。

旧CNN拟合/比较等完成情况见[基线决策](BASELINE_DECISION_V1.md)与[STATUS](STATUS.md)，不能将旧结果算为本轮阶段验收。

## 任务表

| ID | 工作 | 状态 | 产物/完成证据 | 下一动作或阻塞 |
| --- | --- | --- | --- | --- |
| D0 | 总路线与数据/损失/记录细则 | 完成 | [路线图](DATA_FIRST_ROADMAP_V1.md)、[执行细则](DATA_FIRST_EXECUTION_V1.md)；本次已核对模型、损失、mask与旧runner源码 | 文档方案，不表示实现完成 |
| S0-1 | 冻结CNN源码、构造参数、原权重指纹 | 完成 | `data/manifests/data_first_baseline_v1.json`，含源码/协议/数据/三种子检查点SHA256及重载证据 | 不重训；后续实验沿用该参照 |
| S0-2 | 数据版本与政策对应、实验登记模板 | 完成 | 冻结清单的历史边界、joint_split_v2计数和`docs/DATA_FIRST_AUDIT_V1.md` | 旧61条仅历史；clean_release_v1仍待联合审计 |
| S1-1 | 1D/2D/辅助/历史评估联合关系表 | 部分完成（版本回连已修正） | 本地`data/manifests/data_first_relations_v1.jsonl`，214002条；106条split源记录与annotation_review逐条标签匹配；profile/历史别名未完全连接 | profile显式映射、历史测试关系、辅助池关系仍待补 |
| S1-2 | 隔离决定与新开发/测试候选 | 部分完成 | 本地`data/manifests/data_first_admission_v1.jsonl`及`docs/DATA_FIRST_AUDIT_V1.md`；clean_release 74条同序列仅71条标签完全相同，3条标签不一致隔离；全体clean_release不作训练输入 | 需补clean_release母本/构建体和新独立评估候选审核 |
| S2-1 | 待审直接实验2D分层复核 | 完成（不可自动准入） | `data/manifests/joint_quarantine_audit_v1.json`、[审计说明](JOINT_QUARANTINE_AUDIT_V1.md)；70条全部保留隔离 | 新版本需补同源/母本/构建体/primary-study证据；不改当前冻结划分 |
| S2-2 | 一个比较辅助来源的有限试点 | 完成（不可准入） | `data/manifests/data_first_comparative_pilot_v1.json`；固定前20条候选，20/20缺来源与质量核验 | 停止扩大该来源；补齐原始PDB/链、方法/覆盖、映射和隔离证据后另立版本 |
| S3-1 | 辅助配对级mask入口与评价准入 | 未开始 | 待实现与契约检查 | 旧collate不能直接承载positive_only/部分标签；旧evaluate尚未排除无候选监督RNA，须新增准入清单与组级分母检查 |
| S3-2 | 训练样本链路冒烟与重载 | 未开始 | 待生成训练链路报告 | 每条件一个种子最多5更新；B实验2+3、C辅助2+实验3，验证优化器重置和相同后段采样；不访问新测试，不接续冒烟权重 |
| S4 | 正式A/B/C协议冻结 | 未开始 | 待生成data_first_abc_protocol_v1.json | 填U/Ua/Uf/delta/严重退步阈值、长度与分组等 |
| S5-1 | 9次正式运行与全种子开发报告 | 未开始 | 无新运行 | S0–S4通过后才启动 |
| S5-2 | 可用时的新独立测试 | 未开始 | 没有本轮新盲测证据 | 无新独立组时保留未执行及原因，不制造测试集 |
| S6 | 序列预训练/反应性/架构/3D单项分支 | 未开始 | 仅有进入条件和候选方案 | 不与首轮辅助2D叠加，触发后另立协议 |
| S6-L | 固定数据：BCE与BCE＋伙伴竞争损失对照 | 候选实现完成，效果对照未开始 | `src/rna2d/losses.py`提供共享BCE和可选竞争项；`tests/test_pair_losses.py`验证8项契约；基线runner调用共享BCE | 状态：BCE共享实现已采用；竞争项仅候选、契约通过、未接入实验配置、未可选合入、未默认启用。无效果证据，待数据主线归档后冻结协议 |

## 执行日志（只追加）

| 日期 | 任务ID | 动作与证据 | 结果/限制 | 下一步 |
| --- | --- | --- | --- | --- |
| 2026-10-27 | D0 | 阅读路线图、项目约定、基线决策、历史数据报告及当前模型/损失/组批/评价源码；新增执行细则及本表 | 文档已补数据用途、BCE、候选1D损失、A/B/C与续训规则；未修改训练代码、原始数据或权重，未执行新实验 | S0-1，核对基线指纹 |
| 2026-10-27 | D0 / S6-L设计 | 按用户要求新增路线阶段6D及执行细则第8节：伙伴竞争公式、可靠负候选、稳定实现、契约检查、6次独立对照、评价及停止规则 | 仅更新文档；损失实现、lambda/预算冻结及训练均未开始，A/B/C仍固定BCE | 当前仍从S0-1继续，损失分支按进入条件执行 |
| 2026-10-27 | S0-1 / S0-2 / S1-1 | 运行 `scripts/build_data_first_audit.py`，核对CNN源码/协议/训练与留出文件、三种子CNN检查点，并生成基线冻结、联合关系与逐记录去向清单；产物及SHA256见 `docs/DATA_FIRST_AUDIT_V1.md` 与 `docs/DATA_FIRST_AUDIT_V1.json` | S0完成；S1首版机器关系登记完成。joint_split_v2为88/10/8，1D为213473条quarantine；clean_release_v1的141条2D仍training_split_ready=false；未降低准入要求，未启动新训练 | 继续S1-2/S2：补独立实验候选与可追溯比较来源试点；辅助准入和链路通过前不冻结A/B/C |
| 2026-10-27 | S0/S1验证 | `.venv3128/Scripts/python.exe -m unittest discover -s tests -v`；审计清单断言检查 | 82项测试通过；新清单计数/哈希断言通过；未读取旧test作选择，未启动A/B/C | 继续补逐条来源/映射/构建体关系审计；保持当前准入决定 |
| 2026-10-27 | S2-2 | 对`bprna_pdb_candidates_NOT_GOLD.jsonl`按固定文件顺序抽取20条，生成可行性统计和准入决定 | 20/20 `provenance_verified=false`、`quality_verified=false`，原始PDB/链、方法/覆盖或提取策略待核验；结论为stop_and_quarantine，不是训练样本 | 不扩大该来源；补齐证据后另立版本，或归档不可行原因；A/B/C仍不启动 |
| 2026-10-27 | S2-1 | 运行 `scripts/audit_joint_quarantine.py`，审计 `joint_split_v2/quarantine.jsonl` 的证据缺口 | 70/70均为`unverified_homology_or_primary_study`；64条缺homology component、9条study完整性不足、6条缺study ID；决定retain_quarantine，未移动记录 | 新版本补同源/研究/构建体证据；当前数据门禁不变，A/B/C仍不启动 |
| 2026-10-27 | S1版本回连修正 | 根据 `joint_split_v2.json` 的 `paired_source_sha256=244221...` 定位真实来源为`annotation_review_v1/paired.jsonl`；按序列哈希、mask和pairs逐条核对，不再把clean_release的同ID记录直接继承split资格 | 106条源记录精确匹配；clean_release 141条中74条同序列、71条mask/pairs完全匹配、3条标签冲突、67条无同序列对应；训练准入仍只限joint_split_v2 train文件。修正了先前错误地按跨版本ID继承split的关系脚本，并在清单中保留冲突状态 | 重跑审计生成器及契约断言；阶段审计更正需单独commit/push |
| 2026-10-27 | S1版本回连更正 | 按 `joint_split_v2.json` 的源指纹核实真实来源为`annotation_review_v1/paired.jsonl`；生成器改为sequence SHA256关联并逐条比对mask/pairs；补充 clean_release alias/conflict 统计 | 106条source记录精确标签匹配；clean_release 71条精确标签别名不获训练资格，3条同序列异标签 quarantine，67条无split对应；新结构断言通过。更正了前一版错误按跨版本ID继承split的映射 | 全量测试与Git差异检查后，单独commit/push该更正 |
| 2026-10-27 | S1版本更正提交 | Git提交`6de1d82`，推送`origin/main` | 只提交聚合审计、版本映射修正和生成工具；逐样本清单继续被`.gitignore`排除 | 继续S6-L候选实现验证 |
| 2026-10-27 | S6-L协议准备启动 | 确认比较辅助来源试点不可准入，转向路线允许的joint_split_v2固定直接实验数据损失对照；GPU环境为RTX 2080 Ti / torch2.7.1+cu128 / CUDA12.8 | 尚未冻结loss协议、尚未启动训练、不访问test；先只核实train/validation竞争监督适用覆盖 | 生成coverage审计；若监督语义/覆盖足够，再冻结唯一lambda、更新预算、种子及停止门槛 |
| 2026-10-27 | S6-L数据与运行覆盖 | `scripts/audit_pair_loss_coverage.py` 仅读train/validation；创建固定6-run运行器、组均匀采样和checkpoint独立重载工具 | train 88/8组，3130/3130正边端点适用；validation 10/7组，306/306适用；候选负数中位数两边皆16。λ=0.05对应零logit竞争/BCE尺度约0.21。仍未训练或读test；主运行工具未冻结 | 完成契约与格式检查；提交/推送运行工具阶段，再写入正式protocol并提交/推送后才启动六次运行 |
| 2026-10-27 | S6-L候选代码 | 新增 `src/rna2d/losses.py`，把既有正负类平衡BCE抽为共享函数，并实现未启用的可选伙伴竞争loss；`train_baseline.py`和旧比较runner改用同一BCE | 专项契约8项、全套90项测试通过；baseline与controlled-runner `--help` 导入、Python compile、`git diff --check`通过。尚未跑任何效果实验。BCE共享实现是采用的默认实现路径；伙伴竞争仅候选，契约通过但无开发收益证据，未作为可选训练配置、未默认启用 | 保持BCE默认；数据主线/协议通过后再决定是否冻结6次效果对照 |
| 2026-10-27 | D0审阅补充（历史） | 补齐新评价准入待实现状态、无监督RNA/空组分母检查、B/C两阶段短冒烟及试点训练可行性验收 | 文档修改；未实现评价逻辑、未训练或访问测试；该行原先“下一步S0-1”已由后续S0/S1/S2记录完成，不再代表当前续接位置 | 见任务表的S1-2、S2-1、S2-2状态 |
| 2026-10-27 | S6-L工具阶段验证 | `.venv3128/Scripts/python.exe -m unittest discover -s tests -v`、新增脚本`py_compile`、`git diff --check`；修正运行器预检失败日志边界 | 94项测试通过；coverage仅使用joint_split_v2 train/validation，工具静态检查未发现test数据入口；未启动训练、未读取test | 提交并推送覆盖审计与六次运行工具阶段；之后单独冻结并推送正式protocol，再依照明确授权决定是否运行 |
| 2026-10-28 | S6-L评价诊断修复 | 将不受观测真值mask影响的合法候选多伙伴端点率接入共享`evaluate`，加入逐RNA/逐组统计、独立重载核验和4项契约测试；运行器日志改用本地日期 | 98项测试、Python编译和`git diff --check`通过；未知位置仍可进入诊断，F1评价范围未改变；尚未冻结protocol或训练 | 提交并推送诊断修复；随后重新生成并审阅冻结protocol |
| 2026-10-28 | S6-L验证命令失败与纠正 | `python -m unittest tests.test_training_metrics`因`tests`不是可导入包而失败；改用`python -m unittest discover -s tests -v`复跑 | 首次失败仅为测试发现器调用错误，没有执行训练或读取数据；修正调用后98项全部通过 | 保留本次修复并推送；返回S1未完成的profile/历史关系审计，再决定损失协议的冻结时点 |

每次追加：日期/任务ID/实际动作/产物路径与哈希或检查命令/通过或失败/限制/下一步。技术失败不删行。文档检查不能写成模型测试通过。

本次文档核验：PowerShell检查6个变更文档的本地Markdown链接，缺失0项；`git diff --check`通过。首次Python链接检查因当前终端找不到python命令未执行，随后使用PowerShell完成同项检查。未运行模型测试。

审阅补充后的核验：PowerShell检查7个待提交文档的本地文件链接，缺失0项；`git diff --check`通过。仅文档变更，未运行模型测试；提交前另检查暂存差异。

本次版本回连修正核验：`joint_split_v2`源哈希与`annotation_review_v1/paired.jsonl`核对一致；106条split记录按sequence SHA256、observed_mask、pairs精确匹配。clean_release 141条中74条序列相同，71条监督相同，3条监督不同；clean_release全体`training_allowed=false`。修正后生成器断言通过，测试结果见本次S1更正日志。

## 运行登记模板（尚无新运行）

```text
run_id:
condition / seed / parent_retry_run:
protocol_path / protocol_sha256:
source_commit / source_file_sha256:
data_manifest / data_sha256 / isolation_version:
state: 未开始|进行中|阻塞|完成|停止
stage / successful_updates / target_updates:
loss_pair / loss_aux / valid_positive_negative_counts:
checkpoint_path / checkpoint_sha256 / resume_state:
evaluation_split / test_accessed:
result_path / reload_verification:
failure_reason / next_action:
```
