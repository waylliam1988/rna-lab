# 本地数据目录

此仓库提交获取/清洗脚本和数据政策，不直接再分发第三方数据、模型权重或本地运行环境。原始文件、生成的数据与审计报告保存在本地，受.gitignore排除。

- `raw/`：下载归档、官方目录、SHA256与来源快照。
- `processed/`：早期基准清洗结果，不能自动当作已批准实验标签。
- `experimental/`：实验序列/配对候选和探测数据；只有通过溯源与质量审核才可进入训练。
- `comparative/`：独立比较分析辅助候选索引；原文/比对证据未审核的记录不批准训练。
- `manifests/`：可提交的公开来源URL、字节哈希、发布计数与验证摘要；不含第三方序列/信号原始数据。
- `reports/`：本地审计及运行报告。

已有脚本：

```powershell
./run.ps1 scripts/download_data.py
./run.ps1 scripts/prepare_data.py
./run.ps1 scripts/audit_experimental_sources.py --bpseq-zip '你的原始bpRNA归档路径'
./run.ps1 scripts/query_current_experimental_pdb.py
./run.ps1 scripts/screen_probing_samples.py
```

最新获取与清洗范围见 `docs/CLEAN_RELEASE_V1_REPORT.md`、`docs/FINE_CLEANING_PROTOCOL_V2.md`。原始数据、排除原因和待审证据保留；全池没有自动获得训练资格；后续joint_split_v2已对176条扩展2D候选放行88/10/8子集，70条2D与213473条1D继续隔离，1D预训练关闭。旧8条测试已揭盲，后续不得用于调参。见[逐项验证](../docs/STAGED_VALIDATION.md)。探测反应性不等于精确配对标签。
