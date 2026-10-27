# 本地数据目录

此仓库提交获取/清洗脚本和数据政策，不直接再分发第三方数据、模型权重或本地运行环境。原始文件、生成的数据与审计报告保存在本地，受.gitignore排除。

- `raw/`：下载归档、官方目录、SHA256与来源快照。
- `processed/`：早期基准清洗结果，不能自动当作已批准实验标签。
- `experimental/`：实验序列/配对候选和探测数据；只有通过溯源与质量审核才可进入训练。
- `comparative/`：未来的独立比较分析辅助池；当前未建立、未启用。
- `reports/`：本地审计及运行报告。

已有脚本：

```powershell
./run.ps1 scripts/download_data.py
./run.ps1 scripts/prepare_data.py
./run.ps1 scripts/audit_experimental_sources.py --bpseq-zip '你的原始bpRNA归档路径'
./run.ps1 scripts/query_current_experimental_pdb.py
./run.ps1 scripts/screen_probing_samples.py
```

脚本运行会生成本地文件；未通过审计的候选不会自动成为金标准训练集。大规模实验坐标与探测数据的完整获取、注释和质量审计尚未完成。
