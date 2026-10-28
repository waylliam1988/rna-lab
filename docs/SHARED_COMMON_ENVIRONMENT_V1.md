# 公共依赖迁移（2026-10-28）

用户指定：公共包全部安装到系统 Python 3.12.8；发生冲突以 StockAlarm 版本为准。实际固定清单在 `D:/stockalarm/requirements.txt`（位于 StockAlarm Git 仓库的父目录）；仓库 `D:/stockalarm/stockalarm` 未发现自己的 requirements 或虚拟环境。

系统入口：`C:/Program Files/Python312/python.exe`。PyTorch 2.7.1+cu128、CUDA runtime 12.8、NumPy 2.2.6 不变。已有其他项目使用的全局依赖在解析时加入约束，未通过批量升级覆盖它们。全局 pip 26.1.2、pytest 9.0.3、setuptools 70.2.0 沿用迁移前已工作版本；不再保留 RNA 中重复的工具副本。

## 版本决策

| 包 | RNA 迁移前 | 共享版本（StockAlarm） |
| --- | --- | --- |
| joblib | 1.6.0 | 1.5.0 |
| matplotlib | 3.11.2 | 3.10.3 |
| pandas | 2.3.3 | 2.3.1 |
| psutil | 7.2.2 | 7.0.0 |
| python-dateutil | 3.0.0 | 2.9.0.post0 |
| requests | 2.34.2 | 2.32.4 |
| scikit-learn | 1.9.1 | 1.7.0 |
| scipy | 1.18.1 | 1.16.0 |
| tqdm | 4.70.1 | 4.67.1 |
| urllib3 | 2.8.0 | 2.5.0 |

StockAlarm 清单中全部 22 个固定版本纳入共享清单。根据源代码静态导入审核补齐未列入原清单的 pyarrow、astunparse、deap、yfinance、shap；这些版本及传递依赖由 pip 解析后固定。完整公共清单见 `requirements-shared-common.txt`，GPU 基础清单见 `requirements-shared-gpu.txt`。原 `requirements-lock.txt` 作为历史快照保留，不能继续用于覆盖全局环境。

## 安装与边界

```powershell
./scripts/setup_shared_gpu.ps1
./scripts/setup_shared_common.ps1
./scripts/setup_environment.ps1
```

前两个脚本使用系统 Python；最后一个脚本只向 RNA 的共享虚拟环境安装 barnaba、biopython、gemmi、mdtraj 四个专用包。虚拟环境用于保留 RNA 专用包和入口，共享公共包，不复制另一套 Python 或 PyTorch。StockAlarm 可直接使用系统解释器；未修改其业务代码、既有特征文件、模型权重或父目录历史依赖清单。

本次迁移变更了 RNA 的 SciPy/scikit-learn 等版本。代码回归通过不能保证跨版本重现历史数值结果，也不等于重新验证已保存的科学/股票模型；历史实验保留原环境记录，新实验应记录新环境清单。未执行股票数据下载、生产扫描、模型重训或 RNA 科学实验。

## 验证

最终结果见 `data/manifests/shared_common_environment_v1.json`。完整安装前快照、pip 解析/安装报告及执行日志保存在 `C:/ProgramData/RNA-Lab/shared-common-v1/`，不提交二进制包或缓存。

实际验证：90 个公共包的版本和全局来源在系统/RNA 两个入口均通过；60 个模块导入、线性拟合/线性方程、Numba JIT、Feather 读写、Matplotlib 离屏绘图通过；两个入口的三张 GPU 前向/反向/AdamW 均通过；两边 pip check 无冲突；RNA 全部 148 项测试通过；StockAlarm 8 项契约测试中 6 项通过、2 项因没有现有 evolved_gene_pool.json 跳过；合成数据 canonical 特征链路与相对强度/XAI 契约均通过。未使用真实股票数据执行训练或扫描。

已通过 pip 移除正式 RNA 环境的 43 个公共包副本；仅剩 4 个专用包。额外发现的旧 `.venv`（3.12.14/CPU 临时环境）11 个包也已通过 pip 卸载，包括 CPU PyTorch；自动审批拒绝整目录删除，因此仅保留无包的环境空壳目录。正式入口仍是 `.venv3128`。StockAlarm 工作区原有未跟踪 feature_selection/ 保持原样。
