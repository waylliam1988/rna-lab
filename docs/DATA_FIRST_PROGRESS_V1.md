# 数据优先工作包：当前进度与续接记录

更新日期：2026-10-28。范围：总路线图阶段0–6。这里区分旧实验、本次文档设计和未来执行，不设置人员字段。

## 当前停在哪里

**S0基线冻结已完成；S2-1直接实验隔离原因审计完成；S2-2比较来源试点结论为不可准入；profile、历史及本地比较候选精确关系已审计。** clean_release候选关系审计未发现可直接准入的新独立测试；仍需补辅助源逐条来源/映射/隔离审核。旧test已揭盲，不据其调参；A/B/C不启动，直到辅助监督通过门禁且训练/验证链路验收。

旧CNN拟合/比较等完成情况见[基线决策](BASELINE_DECISION_V1.md)与[STATUS](STATUS.md)，不能将旧结果算为本轮阶段验收。

## 任务表

| ID | 工作 | 状态 | 产物/完成证据 | 下一动作或阻塞 |
| --- | --- | --- | --- | --- |
| D0 | 总路线与数据/损失/记录细则 | 完成 | [路线图](DATA_FIRST_ROADMAP_V1.md)、[执行细则](DATA_FIRST_EXECUTION_V1.md)；本次已核对模型、损失、mask与旧runner源码 | 文档方案，不表示实现完成 |
| S0-1 | 冻结CNN源码、构造参数、原权重指纹 | 完成 | `data/manifests/data_first_baseline_v1.json`，含源码/协议/数据/三种子检查点SHA256及重载证据 | 不重训；后续实验沿用该参照 |
| S0-2 | 数据版本与政策对应、实验登记模板 | 完成 | 冻结清单的历史边界、joint_split_v2计数和`docs/DATA_FIRST_AUDIT_V1.md` | 旧61条仅历史；clean_release_v1仍待联合审计 |
| S1-1 | 1D/2D/辅助/历史评估联合关系表 | 部分完成（profile/历史/比较候选精确序列关系已补；生物学来源关系待补） | 本地`data/manifests/data_first_relations_v1.jsonl`，214002条；profile/历史聚合结果见`data/manifests/data_first_profile_history_audit_v1.json`，比较候选聚合结果见`data/manifests/data_first_comparative_candidate_relations_v1.json`；逐记录审计表在本地ignored文件；106条split源记录与annotation_review逐条标签匹配 | 辅助池的原研究、比对、同源/构建体/研究关系仍待逐条补齐；精确序列哈希不替代这些审核 |
| S1-2 | 隔离决定与新开发/测试候选 | 候选审核完成，无新独立测试准入 | 本地`data/manifests/data_first_admission_v1.jsonl`；clean_release关系摘要见`data/manifests/data_first_clean_release_eval_candidate_audit_v1.json`；74条与joint精确同序列，71条标签精确匹配、3条冲突；67条非精确候选含当前split/历史test关系及未解决隔离信息；候选准入为0 | 旧test已揭盲，不重用；继续寻找完整来源、研究、构建体与同源证据的新独立评估组；不足时维持没有新盲测的结论 |
| S2-1 | 待审直接实验2D分层复核 | 完成（不可自动准入） | `data/manifests/joint_quarantine_audit_v1.json`、[审计说明](JOINT_QUARANTINE_AUDIT_V1.md)；70条全部保留隔离 | 新版本需补同源/母本/构建体/primary-study证据；不改当前冻结划分 |
| S2-2 | 一个比较辅助来源的有限试点 | 完成（不可准入） | `data/manifests/data_first_comparative_pilot_v1.json`；固定前20条候选，20/20缺来源与质量核验 | 停止扩大该来源；补齐原始PDB/链、方法/覆盖、映射和隔离证据后另立版本 |
| S3-1 | 辅助配对级mask入口与评价准入 | 未开始 | 待实现与契约检查 | 旧collate不能直接承载positive_only/部分标签；旧evaluate尚未排除无候选监督RNA，须新增准入清单与组级分母检查 |
| S3-2 | 训练样本链路冒烟与重载 | 未开始 | 待生成训练链路报告 | 每条件一个种子最多5更新；B实验2+3、C辅助2+实验3，验证优化器重置和相同后段采样；不访问新测试，不接续冒烟权重 |
| S4 | 正式A/B/C协议冻结 | 未开始 | 待生成data_first_abc_protocol_v1.json | 填U/Ua/Uf/delta/严重退步阈值、长度与分组等 |
| S5-1 | 9次正式运行与全种子开发报告 | 未开始 | 无新运行 | S0–S4通过后才启动 |
| S5-2 | 可用时的新独立测试 | 未开始 | 没有本轮新盲测证据 | 无新独立组时保留未执行及原因，不制造测试集 |
| S6 | 序列预训练/反应性/架构/3D单项分支 | 未开始 | 仅有进入条件和候选方案 | 不与首轮辅助2D叠加，触发后另立协议 |
| S6-L | 固定数据：BCE与BCE＋伙伴竞争损失对照 | v1因验证维度故障停止；评价修复及111项测试通过；v2六次运行全部完成，独立重载通过，汇总结论待执行 | v2协议`data/manifests/data_first_loss_protocol_v2.json` SHA256 `46db4520d0bcfc15319f96def77258b4c62ec612837d34caa2a281c098a02659`；六个checkpoint均独立重载，`data/manifests/data_first_loss_verification_v2.json` SHA256 `ca1117899d08c0bad03871688924d6ab9e1c9a51645827b01df55daa67d311ef` | 候选状态保留；开发门槛待汇总判定；未可选合入；默认仍为BCE。运行指标只作预声明validation开发判定，不访问test |

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
| 2026-10-28 | S1 profile/历史关系审计工具 | 新增只读精确关系审计`scripts/audit_profile_and_history_relations.py`；要求profile数组/mask、SQLite、1D隔离表、RMDB来源收据与本地原文件哈希一致；历史split只按精确序列回连，test/test_external标成既往暴露；新增3项关系分类契约测试 | 定向测试3项及全套101项测试通过，脚本编译、忽略规则和`git diff --check`通过；尚未运行全量数据审计；明细明确留本地且training_allowed恒false | 提交/推送工具阶段后运行实际profile与历史关系核验；不访问模型预测或启动训练 |
| 2026-10-28 | S1 profile/历史关系审计准备 | 核对运行输入为`probing_clean_v2/profiles.sqlite3`、167个含profile的RMDB原文件、`joint_split_v2/joint_1d_assignments.jsonl`及当前/历史JSONL；扩展脚本验证clean-release与joint 1D赋值完整哈希集合相同，并仅输出聚合关系数 | 细化后的审计尚未启动；只读校验原始字节SHA256/profile数组/mask/精确序列关系；预计核验约12.2GB；不读取模型预测或改数据 | 完成脚本检查并推送后，另记实际审计启动状态 |
| 2026-10-28 | S1 profile/历史关系审计术语与指纹校正 | 全量测试审查发现profile与joint 1D赋值关系被误称为paired-record别名；改为明确的1D赋值关系，并补充clean-release序列自身SHA256复核；移除易被误读的“未成对profile可继承paired eligibility”措辞 | 尚待本次回归验证；未运行全量数据审计、训练或模型评价；原始数据和准入状态未变 | 回归通过后提交/推送审计工具，再执行只读数据核验 |
| 2026-10-28 | S1 profile/历史关系审计执行 | 只读运行`scripts/audit_profile_and_history_relations.py`；提交聚合报告`data/manifests/data_first_profile_history_audit_v1.json`（SHA256 `ac3f0c88678912aeff21255752e2362b74cd78cf5c918e7b1a402f3541fc8c32`）；本地profile/history明细SHA256分别为`a04ca6cdad8cfee9a30a6ded8e4367d5fc18509e6a51d03760c393a8fed051f1`和`e58f39073edef5f5c1d05159f753f67317f35740de0a8e0e62066a802aff91a5` | 验证344495个实测profile/212652条唯一序列；167/167来源文件、共12194330801字节本地SHA256与收据一致，且167条上游SHA也已核验；213579条clean-release序列与1D赋值集合和内容哈希一致。所有profile序列均属quarantine，pair label为0、training_allowed=false。历史JSONL共18636行；旧archiveII有7条精确序列对应当前train但pair标签冲突；paired_v0_2 train/validation/test分别有24/5/3条当前配对别名，其中24/5条标签冲突、test别名中1条mask/pairs相同而2条冲突。所有逐记录表被Git忽略；历史test/test_external仅记暴露，不用于新训练/选择。未读模型预测、未训练、未改标签/划分；精确哈希审计不能替代同源/构建体/研究审核，也未建立新独立测试 | 提交并推送聚合结果与台账；辅助源关系和生物学隔离仍按原准入门禁审核 |
| 2026-10-28 | S1 比较候选全池精确关系审计准备 | 基于固定20条试点复核本地669条`bprna_pdb_candidates_NOT_GOLD.jsonl`、477条唯一序列和现存bpRNA压缩包；发现试点20条中4条已与当前joint train/validation精确重合；新增逐条压缩包重建与joint/历史精确别名审计脚本及3项边集契约测试 | 全套104项测试、脚本编译和`git diff --check`通过；审计尚未运行；来源所有权、质量/标签、PDB链映射和准入均不因精确别名审计而改变 | 提交/推送工具后执行全池只读审计；仍不扩大来源、不训练或用候选作新测试 |
| 2026-10-28 | S1 比较候选审计首次运行失败与输入契约修正 | 全池审计在发布报告前发现28/669条候选含`N/X/P/I`非ACGU字符（其余641条ACGU-only）；原候选索引保留这些字符，不能静默替换；工具改为记录字母表分层并继续做原始序列/边集核验，新增契约测试 | 首次运行仅因过严字母表断言失败，未写出报告、未改候选或压缩包；非ACGU记录继续不具训练/评估资格，不能把BPSEQ边集当真值 | 回归通过后重新执行全池只读审计，并记录非ACGU数量与限制 |
| 2026-10-28 | S1 比较候选修正后测试命令失败与纠正 | 定向命令`python -m unittest tests/test_comparative_candidate_relations.py -v`未按文件路径发现测试；改用`python -m unittest discover -s tests -p test_comparative_candidate_relations.py -v` | 首次失败发生在导入发现阶段，没有执行测试；代码/数据未改动，随后按正确发现器调用重跑 | 保留失败记录并完成定向及全套回归后再提交修正 |
| 2026-10-28 | S1 比较候选审计输入契约修正验收 | 定向4项测试、全套105项测试、Python编译与`git diff --check`全部通过；在报告发布前校验joint_split_v2冻结文件指纹/1D计数、压缩包条目数及候选数据与原始BPSEQ重建一致 | 28条非ACGU记录被显式分层，不改写序列；仍未执行完整关系审计、训练或模型评价 | 提交/推送工具修正后执行全池只读审计 |
| 2026-10-28 | S1 比较候选全池精确关系审计执行 | 只读运行`scripts/audit_comparative_candidate_relations.py`；聚合报告`data/manifests/data_first_comparative_candidate_relations_v1.json`（SHA256 `c79f1d42dc7b09dce12923daf6a359c39841326cb95720d3ca763e4bdc07b935`）；候选源SHA256 `d96db987c0b1fcf8d296f0e5f629cba477fafdd713271ddd2c992d4cf95cb4fb`；本地ZIP SHA256 `14e2824f50072c8c8bfe99ed06e19d92f72a79112c6cbdd5e08d8fa51dbf96d0` | 669/669候选成员成功重建，477条唯一序列；28条记录（25条唯一序列）含非ACGU；重复序列记录192条，48条序列含不同候选边集；所有669条来源/质量/辅助准入标记均false。与joint 1D精确别名记录为train 41、validation 7、quarantine 56、未映射565；与当前paired记录精确序列别名为train 41、validation 7，边集均不同。历史archiveII/paired_v0_2 train/validation/test候选别名分别为15/43/13/5条，均边集不同；test/test_external未发现候选别名。候选训练和新测试资格均为0；未读预测、未训练、未改标签/划分 | 该来源仍停止扩大并隔离；下一步只能补原始PDB/链、URL/下载收据、文献、方法/覆盖、提取策略、比对映射及同源/研究/构建体证据，另立版本后再审 |
| 2026-10-28 | S1-2 clean_release新独立评估候选准备 | clean_release 141条中74条与当前joint数据精确同序列，67条非精确别名；本地RNA3DB精确序列对照仅5条可提供非占位同源component，67条均无显式construct provenance，6条缺主要文献引用；用joint构建器同一study/组件/PDB/相似性规则做的初步关系图显示67条中14条连接当前train、1条连接当前test | 初步审阅未开旧测试预测或模型分数；67条不能仅凭非精确别名称为独立组，候选仍未准入。现新增可复现聚合审计工具，以冻结源指纹复核关系图并核对旧test精确别名 | 验证并推送工具阶段，再执行只读关系审计；独立组不足则保持无新盲测 |
| 2026-10-28 | S1-2 clean_release候选审计工具阶段 | 新增`scripts/audit_clean_release_eval_candidates.py`和5项准入/并查集关系测试；复用冻结split构建器的RNA3DB、primary citation、PDB实体和identity/coverage连接规则；仅输出聚合关系和数据指纹 | 定向5项、全套110项测试、Python编译及`git diff --check`通过；尚未运行实际关系图，报告只会标记候选，不会生成或准入新测试 | 提交/推送工具后执行只读审计；再根据聚合连接/缺失证据给出是否存在可独立候选的结论 |
| 2026-10-28 | S1-2首次clean_release报告口径失败与修正 | 报告复核发现“非精确候选连到current split”汇总字段误计入74条精确别名；纠正为只统计67条非精确候选，并追加74条精确别名按split及mask/pairs结果的分层统计 | 初次生成的聚合报告未提交且已删除重建；关系组判定逻辑和0条准入结论未变；修正后定向5项测试、编译和`git diff --check`通过，完整审计待重新执行 | 完成全套回归后推送统计口径修正并重新运行只读审计 |
| 2026-10-28 | S1-2 clean_release独立评估候选审计执行 | 运行`scripts/audit_clean_release_eval_candidates.py`；聚合报告`data/manifests/data_first_clean_release_eval_candidate_audit_v1.json`（SHA256 `8f7c4f717d8262117db482e4390efba26b0f223da81d255c3e34ef5690569814`） | 141条候选中74条与joint精确同序列（train/validation/test=59/8/7），其中71条mask/pairs一致、3条冲突；其余67条中14条与当前train关联、1条与当前test关联；clean_release全体中另有2条与历史paired_v0_2 test精确同序列。67条非精确记录并为42组：1组连train、1组连current test、1组历史test暴露、39组缺完整独立性证据。141条直接来源/质量标记通过且无模型预测，但显式construct provenance为0/141，8条缺primary citation；上下文127/141为complex。新独立测试准入0；未读模型预测/旧测试成绩、未训练、未改原始标签/划分 | 不从clean_release 141条直接抽名义测试；后续需逐条补母本/构建体及家族/研究关系和实验背景，再另立版本；没有新独立组时保持无新盲测 |
| 2026-10-28 | S6-L正式协议冻结 | 运行`scripts/freeze_pair_loss_protocol.py`，生成`data/manifests/data_first_loss_protocol_v1.json`（SHA256 `9c51778c2f11abbdbcd79df8cfce252f6e8ef94afbe2131af3af91c56c684a56`）；复核覆盖审计、数据与源码指纹、每种子初始化和共享采样日程 | 协议状态为`frozen_before_training`；lambda=0.05，seed 42/43/44，BCE与BCE＋竞争项共6次、每次550次更新；固定最终检查点与0.5阈值，预声明开发门槛及失败规则。110项测试通过，Python编译、`git diff --check`通过；CUDA可用（RTX 2080 Ti）；未启动正式训练或访问test | 单独提交并推送冻结协议阶段；之后只运行预声明6次实验，不追加种子、不调lambda/阈值、不按validation选checkpoint |

| 2026-10-28 | S6-L运行启动 | `bce` seed42；protocol `9c51778c2f11abbdbcd79df8cfce252f6e8ef94afbe2131af3af91c56c684a56`；train `4168f00f61d9a3330db33b45308fe247c6ba15fd1c50ce4b868559b78f3ffcb9`；validation `794e1cd6692eccfbad1212fa8e4246c2698eaede8fb686f8a502e52908451a71`；550更新 | 仅加载train/validation；检查点本地保存，test未打开 | 等待运行完成或保留异常 |
| 2026-10-28 | S6-L运行失败 | `bce` seed42；protocol `9c51778c2f11abbdbcd79df8cfce252f6e8ef94afbe2131af3af91c56c684a56`；异常 `ValueError: Scores must be a non-empty square matrix matching the sequence` | 失败运行目录保留，未触碰test；按冻结规则不自动重试 | 保留失败原因；该条件/种子不重跑 |
| 2026-10-28 | S6-L停止判定 | seed42已完成550次更新并写出checkpoint（SHA256 `1c356a7c230e5683ddea35e36341a0fd84f6cd91ca5ca52b5327ab61c8aa7536`）；验证阶段短序列与批次补齐分数矩阵尺寸不匹配，未写出`result.json` | 已确认是evaluation诊断输入长度契约问题；冻结协议要求不重试或改阈值/协议，故不启动其余5次（同一评价代码会触发相同技术失败）。未访问test，不能给出效果结论 | 提交并推送失败留痕；后续若修复评价代码，需另立协议版本，不能补跑本版 |
| 2026-10-28 | S6-L共享评价变长批次修复 | `evaluate`对模型输出逐条裁剪到真实序列长度，再执行解码和多伙伴诊断；新增`test_evaluation_crops_batch_padding_per_sequence` | 定向评价测试5项通过，全套111项测试通过；`py_compile`及`git diff --check`通过。仅测试合成短/长序列，未读取test、未重跑v1、未触碰失败checkpoint；v1仍停止，历史错误留痕保留 | 提交并推送正确性修复；再为相同预声明单因素比较建立全新协议版本，六次运行均从头开始 |
| 2026-10-28 | S6-L v2运行工具版本化 | 冻结脚本支持只生成v2；训练、独立核验和汇总脚本接受显式协议路径，并按版本隔离runs及manifest；执行细则记录v1失败和v2重启边界 | 111项测试、脚本帮助/导入、Python编译和`git diff --check`通过；v2流程沿用已冻结的数据/损失/预算/门槛，不读取test，不复用v1 checkpoint/指标 | 提交并推送版本化工具；随后冻结v2协议及源码指纹，再启动六次正式配对运行 |
| 2026-10-28 | S6-L v2协议冻结 | 运行`scripts/freeze_pair_loss_protocol.py --version v2`，生成`data/manifests/data_first_loss_protocol_v2.json`（SHA256 `46db4520d0bcfc15319f96def77258b4c62ec612837d34caa2a281c098a02659`）；核对覆盖审计、train/validation、模型/损失/训练/采样及运行工具源码指纹 | `frozen_before_training`；仍为lambda=0.05、seed 42/43/44、每条件550更新、共6次；v2明确不复用v1 checkpoint/指标。数据与10个源码指纹全部匹配，test访问仍禁止；未启动训练 | 提交并推送v2协议阶段；之后按冻结顺序启动六次运行，每个条件/seed仅一次 |
| 2026-10-28 | S6-L运行启动 | `bce` seed42；protocol `46db4520d0bcfc15319f96def77258b4c62ec612837d34caa2a281c098a02659`；train `4168f00f61d9a3330db33b45308fe247c6ba15fd1c50ce4b868559b78f3ffcb9`；validation `794e1cd6692eccfbad1212fa8e4246c2698eaede8fb686f8a502e52908451a71`；550更新 | 仅加载train/validation；检查点本地保存，test未打开 | 等待运行完成或保留异常 |
| 2026-10-28 | S6-L运行完成 | `bce` seed42；550更新；schedule `0436b53fd3d29851eb43bc13f571a6480f3c42727c4df159dd12f0bab5b189db`；checkpoint `4f7ebc61c361dcd30859918f3ed13f64ba6aa5dcbe2e4fec6027be09c8c00037` | validation group-macro F1=0.511988；test未访问；独立重载待统一执行 | 继续冻结名单中的其他条件/种子，不改协议 |
| 2026-10-28 | S6-L运行启动 | `bce` seed43；protocol `46db4520d0bcfc15319f96def77258b4c62ec612837d34caa2a281c098a02659`；train `4168f00f61d9a3330db33b45308fe247c6ba15fd1c50ce4b868559b78f3ffcb9`；validation `794e1cd6692eccfbad1212fa8e4246c2698eaede8fb686f8a502e52908451a71`；550更新 | 仅加载train/validation；检查点本地保存，test未打开 | 等待运行完成或保留异常 |
| 2026-10-28 | S6-L运行完成 | `bce` seed43；550更新；schedule `2b12cb70581f6ef836bfc9035d7a3170219f05a937b1723096480b6bd9e31cbb`；checkpoint `527f3f291f8854f954a0ab3909c9a91dcaf1c91c85c4866a7124f423d7444fff` | validation group-macro F1=0.546002；test未访问；独立重载待统一执行 | 继续冻结名单中的其他条件/种子，不改协议 |
| 2026-10-28 | S6-L运行启动 | `bce` seed44；protocol `46db4520d0bcfc15319f96def77258b4c62ec612837d34caa2a281c098a02659`；train `4168f00f61d9a3330db33b45308fe247c6ba15fd1c50ce4b868559b78f3ffcb9`；validation `794e1cd6692eccfbad1212fa8e4246c2698eaede8fb686f8a502e52908451a71`；550更新 | 仅加载train/validation；检查点本地保存，test未打开 | 等待运行完成或保留异常 |
| 2026-10-28 | S6-L运行完成 | `bce` seed44；550更新；schedule `c06a3efc718e08a927d4cfba37279a0cc31a8535bdbb6c86fd5a6ed1450a6269`；checkpoint `1b03b2a5d8e8458fa336de57ef0d606ba0f6204717f83793609c1080e1257a16` | validation group-macro F1=0.509172；test未访问；独立重载待统一执行 | 继续冻结名单中的其他条件/种子，不改协议 |
| 2026-10-28 | S6-L运行启动 | `bce_competition` seed42；protocol `46db4520d0bcfc15319f96def77258b4c62ec612837d34caa2a281c098a02659`；train `4168f00f61d9a3330db33b45308fe247c6ba15fd1c50ce4b868559b78f3ffcb9`；validation `794e1cd6692eccfbad1212fa8e4246c2698eaede8fb686f8a502e52908451a71`；550更新 | 仅加载train/validation；检查点本地保存，test未打开 | 等待运行完成或保留异常 |
| 2026-10-28 | S6-L运行完成 | `bce_competition` seed42；550更新；schedule `0436b53fd3d29851eb43bc13f571a6480f3c42727c4df159dd12f0bab5b189db`；checkpoint `f63e3b101c4e7c8f3d59622e4eb03ee30e869ef2fcc43b8adbfd5803accce021` | validation group-macro F1=0.527559；test未访问；独立重载待统一执行 | 继续冻结名单中的其他条件/种子，不改协议 |
| 2026-10-28 | S6-L运行启动 | `bce_competition` seed43；protocol `46db4520d0bcfc15319f96def77258b4c62ec612837d34caa2a281c098a02659`；train `4168f00f61d9a3330db33b45308fe247c6ba15fd1c50ce4b868559b78f3ffcb9`；validation `794e1cd6692eccfbad1212fa8e4246c2698eaede8fb686f8a502e52908451a71`；550更新 | 仅加载train/validation；检查点本地保存，test未打开 | 等待运行完成或保留异常 |
| 2026-10-28 | S6-L运行完成 | `bce_competition` seed43；550更新；schedule `2b12cb70581f6ef836bfc9035d7a3170219f05a937b1723096480b6bd9e31cbb`；checkpoint `6515470321d9ba35264238d9e47452ef505d534f50a84810b7d81ec33090d096` | validation group-macro F1=0.554906；test未访问；独立重载待统一执行 | 继续冻结名单中的其他条件/种子，不改协议 |
| 2026-10-28 | S6-L运行启动 | `bce_competition` seed44；protocol `46db4520d0bcfc15319f96def77258b4c62ec612837d34caa2a281c098a02659`；train `4168f00f61d9a3330db33b45308fe247c6ba15fd1c50ce4b868559b78f3ffcb9`；validation `794e1cd6692eccfbad1212fa8e4246c2698eaede8fb686f8a502e52908451a71`；550更新 | 仅加载train/validation；检查点本地保存，test未打开 | 等待运行完成或保留异常 |
| 2026-10-28 | S6-L运行完成 | `bce_competition` seed44；550更新；schedule `c06a3efc718e08a927d4cfba37279a0cc31a8535bdbb6c86fd5a6ed1450a6269`；checkpoint `1422417c0209418da57fb3935481ca25679258a583886ae9cf0107742c595d7a` | validation group-macro F1=0.491957；test未访问；独立重载待统一执行 | 继续冻结名单中的其他条件/种子，不改协议 |
| 2026-10-28 | S6-L v2六次正式运行完成 | 严格按顺序完成`bce`及`bce_competition`的seed 42/43/44；每次550更新，从协议初始化状态开始，使用共同采样日程，validation在最终checkpoint评价 | 6/6运行成功，结果/step日志/checkpoint保留在本地ignored `runs/data_first_loss_v2/`；各结果checkpoint SHA256与文件一致。validation group-macro F1：BCE 0.511988/0.546002/0.509172；competition 0.527559/0.554906/0.491957。未访问test；当前只列运行结果，不据单次结果停止或改门槛 | 独立核验全部六个checkpoint的验证预测后，再单独执行预声明门槛汇总 |
| 2026-10-28 | S6-L v2独立checkpoint核验 | 运行`scripts/verify_pair_loss_comparison.py --protocol data/manifests/data_first_loss_protocol_v2.json`，生成`data/manifests/data_first_loss_verification_v2.json`（SHA256 `ca1117899d08c0bad03871688924d6ab9e1c9a51645827b01df55daa67d311ef`） | 6/6检查点严格加载成功；validation group-macro、逐RNA多伙伴诊断、逐组多伙伴诊断及解码预测均与run结果重现；全部`test_accessed=false`。仅使用冻结validation，不进行选择或测试评估 | 提交并推送验证产物与台账；之后运行固定summarizer应用冻结gate |
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
