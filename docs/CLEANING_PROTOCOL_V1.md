# 实验坐标清洗与配对数据 v0.1 协议

## 两个层次

全量坐标审计保留所有长度和背景；首版可训练配对集另限定范围。超出首版范围不是低质量的同义词。

### 全量坐标审计

- 读取本地PDB_RNA.zip，不批量解压，不覆盖原始文件。记录ZIP目录指纹、CRC、每份CIF的SHA256和代码指纹；支持同配置断点续跑。
- 使用Gemmi 0.7.5解析mmCIF，区分实体ID、label_asym_id、auth_asym_id和残基序号。
- 对照冻结的RCSB实验条目目录，排除理论模型/不匹配条目；纯DNA不冒充RNA。
- X-ray≤3.0Å、EM≤3.5Å仅为初筛；NMR单独待约束与系综质量审核，不用X-ray阈值粗暴判坏。
- 每个位点要求完整标准碱基重原子和C1'，占据率≥0.9；不跨替代构象拼出完整碱基。缺失/低占据率位置的observed_mask为false。
- 完整碱基覆盖率≥95%才通过本轮坐标初筛。修饰或微异质位置保留但进入专门待审池。
- 不以B因子直接代表柔性或可靠度；全局分辨率也不能代表局部电子密度质量。

### 首版实验配对集

- 先处理20–512nt、未修饰、完整碱基坐标、单条RNA、无蛋白/DNA背景的保守子集。保留其他结构候选，不截断并冒充完整RNA。
- 从RCSB官方HTTPS重新取得入选的小型CIF，记录下载URL、时间、SHA256；不靠本地归档文件名证明实验溯源。
- Barnaba 0.1.9在实验坐标上做几何注释。只使用annotate_traj，不运行热力学折叠或AI预测。适配器显式处理Å→nm与残基索引映射。
- 标准配对任务只收严格WCc/GUc、AU/GC/GU、索引距离≥4的配对。非经典/弱相互作用保留在原始注释中，相关位置遮罩；不将其静默改为未配对。
- 若出现多个标准配对伙伴，不用贪心选一个当真值，隔离整条记录。
- 同一序列的多个结构在共同可观测位置上若拓扑冲突，进入构象/条件冲突池；一致时取分辨率较优的代表并保留全部来源，掩码取交集。
- 首版保留至少50%位置有明确监督、且至少存在一个明确标准配对的记录；这是标签可用度范围参数，不把真实非经典接触当低质量。训练和评估都尊重掩码；评估先独立解码，再忽略未知端点的预测，不利用真值掩码改变解码选择。因此报告的是可评估位置上的F1，不冒充整条RNA完整真值F1。
- 只有通过此协议的记录才标记为experimental_coordinate_annotation。quality_verified表示通过公开的本轮筛选协议，**不表示标签绝无误差或全部生理条件下成立**。

## 去重与划分

使用RNA3DB 2026-01-05中已分配的同源连通分量，加上≥80%全局序列一致性关系，重新合并为不跨集合的组。相同序列只保留一个一致标签代表。

全局比对固定match=2/mismatch=-1/open=-3/extend=-0.5；identity=identities/(identities+mismatches+gaps)。长度比小于0.8不做该全局80%比较；远缘/片段同源性仍依赖已有同源分量，不能声称排除了所有远缘结构同源。

没有已知RNA3DB分量或只有component_0的序列，单独输出unknown_homology_NOT_FOR_TRAINING，不擅自随机放入训练。固定种子42，按组大小从大到小、优先填补样本量目标缺口，接近80/10/10；不拆同源组。初始仅按组数量的划分会造成样本不均，保留为paired_v0_1草稿；最终版本为paired_v0_2。只依据样本规模修正规则，未查看模型测试成绩。

## 比较分析辅助池

单独为101649条非PDB来源记录建立待审核索引。必须进一步确认原始条目、文献、validation technique不是predicted、比对映射和同源性；当前不自动放行，也不把它们统一污名化为模型伪标签。

## 明确局限

首版使用一个公开几何注释器，尚无独立多工具共识与局部密度验证。它是保守范围内可复现的数据版本，不是覆盖所有RNA的终极金标准。非经典、多链、蛋白/配体依赖、修饰、NMR、共转录和探测反应性需要分别扩展。

Barnaba论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC6348988/ 。该工具是GPLv3外部依赖，本项目不复制或改写其科学算法。

## 运行

```powershell
./run.ps1 -m pip install -r requirements-data.txt
# Windows如果缺少原生运行时，可使用已验证签名的本地DLL源，或安装微软官方VC++运行时
./scripts/prepare_native_runtime.ps1
./run.ps1 scripts/clean_coordinate_archive.py --workers 2
./run.ps1 scripts/build_experimental_pairs.py
./run.ps1 scripts/index_comparative_candidates.py
```

运行结果保存在data/experimental和data/comparative，原始数据与生成的数据均不提交GitHub。RNApolis本轮因Windows下mmcif编译依赖未安装成功；实际标注方案选择Barnaba，并将NumPy固定为2.2.6以兼容其旧版字符串数组API，未修改第三方算法。
