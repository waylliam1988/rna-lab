# 实验 RNA 来源扩充 v1

本轮先收集可追溯的原始证据，再审核可训练记录。所有新增下载收据都标记 `quality_approved=false`。不改变现有 paired_v0_2 训练/验证/测试数据。

## 来源与边界

| 来源 | 收集内容 | 使用边界 |
| --- | --- | --- |
| RCSB PDB | 冻结查询中的 10,522 个实验 RNA 条目；复用已有大归档与已验证 CIF，只下载缺失的 1,225 个 CIF | 条目数不等于独立序列数；仍需局部坐标、修饰、链间接触与复合物背景审查 |
| RMDB | 官方 data-* release 的 1,024 个 RDAT 资产 | 原始实验探测容器；STRUCTURE 字段不能直接作实验配对真值；需审核试剂、SEQPOS、突变/序列映射、误差与覆盖度 |
| RNAsolo | BGSU 代表/成员及 Rfam 结构片段的 FASTA、BPSEQ、DBN、清单，共 12 个包 | 坐标衍生注释；不同视图重叠，多链边界、残基映射与注释策略仍待核验 |
| RASP v2 | 676 个 raw 展示记录对应 508 个去重下载 URL；94 个接触数据文件 | 排除目录中的 118 个 imputed 轨道。保留重复 URL 对应的所有链向/条件元数据；接触证据不等于精确碱基对 |
| FR3D | 官方全量相互作用注释压缩表 | 供独立几何交叉检查；以服务器 Last-Modified 为准，不能当作当前 PDB 的全覆盖注释 |
| RASP 参考文件 | 源站下载页的 48 个 FASTA/GFF 链接 | 用于探测信号的序列和基因组位置映射；参考序列/基因注释不是实验配对标签 |

源站：<https://www.rcsb.org/>、<https://rmdb.stanford.edu/about/>、<https://rnasolo.cs.put.poznan.pl/>、<https://rasp2.zhanglab.net/download/>、<https://www.bgsu.edu/research/rna/help/rna-3d-hub-help/accessing-rna-3d-hub-data.html>。

`data/raw/experimental_acquisition_v1/plan.json` 和 `data/raw/additional_experimental_acquisition_v1/plan.json` 是冻结计划，原始目录 JSON/上游版本另行保存。下载使用证书校验、大小/SHA256 校验、压缩格式检查；ZIP/TAR 不自动解压。上游提供 SHA256 时另行比对，不提供时本地 SHA256 只保证本地可追溯。

FR3D 若 requests 遇到 TLS EOF，可用系统 curl 默认 TLS 校验下载到 `.curl.partial`，成功退出后运行 `scripts/finalize_fr3d_curl.py` 校验并记录收据。不能关闭证书校验。

FR3D 已通过该方式下载，响应 `Last-Modified: Wed, 20 Nov 2024 07:45:58 GMT`；该历史注释快照不能宣称覆盖全部当前 PDB。附加包中两个 BED 接口持续返回 HTTP 500：小鼠 Keth-seq 的一个双链向文件、HIV-1 SHAPE 的一个文件。原失败保留在报告中，另用 `scripts/acquire_rasp_fallbacks.py` 从源站下载了对应的三个 raw BigWig 轨道包，见 `data/reports/rasp_bigwig_fallback_v1.json`；不是将失败 BED 偷换为成功。

## 运行与复核

```powershell
.\.venv3128\Scripts\python.exe -X utf8 scripts/acquire_experimental_corpus.py --workers 6
.\.venv3128\Scripts\python.exe -X utf8 scripts/catalog_additional_experimental_sources.py
.\.venv3128\Scripts\python.exe -X utf8 scripts/acquire_additional_sources.py --workers 4
.\.venv3128\Scripts\python.exe -X utf8 scripts/acquire_rasp_references.py
.\.venv3128\Scripts\python.exe -X utf8 scripts/acquire_rasp_fallbacks.py
.\.venv3128\Scripts\python.exe -X utf8 scripts/index_acquired_rnasolo.py
.\.venv3128\Scripts\python.exe -X utf8 scripts/summarize_acquisition.py
```

`--retry-failed` 仅用于前一轮已结束且冻结计划未改变的情况；复用已验证下载，重试未通过项。重试报告不代表重新对所有已验证文件做完整哈希扫描。

下载进度/失败详情在 `data/reports/experimental_acquisition_v1_all.json` 和 `data/reports/additional_experimental_acquisition_v1.json`。汇总在 `data/reports/internet_acquisition_summary_v1.json`，候选序列/配对索引在 `data/experimental/internet_annotation_candidates_v1/candidates.sqlite3`。

## 下一轮清洗

1. 审核链内/链间配对及复合物背景，不能把链外配对端点标为链内负例。
2. 处理修饰碱基与完整/缺失残基映射，并用独立几何注释交叉检查；冲突与未知标签保留遮罩。
3. 对探测信号按原始文献、实验条件、试剂选择性、覆盖度和误差审核；保留为软证据，不凭空生成配对真值。
4. 以同源簇隔离再划分，比较分析数据另建辅助池；报告逐项剔除原因与有效样本量。

本轮不盲目下载 SRA/GEO 中所有测序 reads，也没有逐篇穷尽互联网散落附件；因此不能称为全球实验数据的绝对上限。这里获得的是已确认公开目录的版本化覆盖，后续新增来源应建立新版本。

RSVdb 的两个文献域名 `taolab.nwafu.edu.cn` / `taolab.nwsuaf.edu.cn` 在 2026-10-27 实测均返回 HTTP 503，服务器要求 43,200 秒后重试；未将错误页面保存为数据，也没有宣称已覆盖该数据库。

## RMDB 容器盘点

`scripts/audit_rdat_containers.py` 对已验证的 1,024 个文件按原始 ASCII 字段流式计数，结果在 `data/reports/rdat_container_inventory_v1.json`。共有 4,545,388 个 REACTIVITY 行；4,545,387 个 ANNOTATION_DATA 行中，3,401,751 行含源站声明的 `warning:badQuality`，2,741,573 行声明弱信噪比；这些集合重叠，不能相加。大文件数量不能当成有效实验样本数。

这是原始容器盘点，还没有批准任何探测记录入训练。后续必须逐条排除/隔离坏质量标记，并审核未标记记录；不能认为没有 warning 就质量合格。历史 RDAT 注释中存在非 UTF-8 文本，本轮仅按原始字节识别 ASCII 字段，不破坏原文件。序列注释、反应性行数的差异需要核验关联与重复情况。
