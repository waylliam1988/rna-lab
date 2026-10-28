# 本地环境

## 公共依赖共享（2026-10-28，当前配置）

热力学基线新增全局ViennaRNA==2.7.2（共享清单现91项）；不修改既有Torch/NumPy/StockAlarm版本。此前90项迁移manifest是安装当时快照。

用户要求公共依赖全局安装，版本冲突按 StockAlarm 的 `D:/stockalarm/requirements.txt` 处理。公共清单为 `requirements-shared-common.txt`；SciPy 1.16.0、scikit-learn 1.7.0、Pandas 2.3.1、Matplotlib 3.10.3 等采用 StockAlarm 版本。Python 3.12.8、PyTorch 2.7.1+cu128、CUDA runtime 12.8、NumPy 2.2.6 不变。

RNA 的 `.venv3128` 共享系统公共包，仅保留 barnaba、biopython、gemmi、mdtraj 四个 RNA 专用包。全局包来源与版本由 `scripts/check_shared_common.py` 检查，安装使用 `scripts/setup_shared_common.ps1`。`requirements-lock.txt` 是迁移前历史快照，不代表当前公共环境。完整版本决策、安装与验证见 [公共依赖迁移记录](SHARED_COMMON_ENVIRONMENT_V1.md)。以下共享 GPU 记录描述前一阶段，47 个本地包已不再是当前状态。

## 系统Python与共享GPU环境（2026-10-28，前一阶段）

本机已安装系统级Python3.12.8：`C:/Program Files/Python312/python.exe`。RNA虚拟环境原先依赖的`.runtime/python3128/python.exe`已经不存在；本次已备份pyvenv.cfg及venv启动器，并以系统3.12.8重绑定`.venv3128`，无需重建项目依赖。

按用户要求，全局GPU基础版本固定与RNA相同：torch2.7.1+cu128、CUDA runtime12.8、cuDNN9.7.1、NumPy2.2.6。共享依赖清单为requirements-shared-gpu.txt；原requirements-lock.txt保留历史完整项目锁定。CUDA驱动为系统级，PyTorch运行时来自官方wheel；本次不运行CUDA/驱动安装包、不安装下载目录中不同版本的cuDNN压缩包。

安装包现位于`D:/E backup/下载/torch-2.7.1+cu128-cp312-cp312-win_amd64.whl`，经大小与官方SHA256核验后，原C盘wheelhouse和pip缓存两个硬链接均已移除。全局已安装的PyTorch未修改。setup_shared_gpu.ps1优先使用该本地文件并核对SHA，可用-WheelDirectory指定其他目录；安装/GPU日志仍在C:/ProgramData/RNA-Lab/shared-python3128/。位置变更见data/manifests/shared_pytorch_wheel_relocation_v1.json；原wheel_provenance_v1.json保留为安装当时的历史记录。

复用本地3273024349字节torch wheel，SHA256与官方cu128索引核对一致（2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30）；包缓存与安装/GPU检查记录位于`C:/ProgramData/RNA-Lab/shared-python3128/`。全局和RNA共享入口均通过三张GPU前向/反向/AdamW实测，版本及来源一致，148项项目测试和两边pip check通过。结果见data/manifests/shared_python_gpu_environment_v1.json。全局验证后已卸载项目torch重复副本，当时其余47个项目包的METADATA哈希不变；随后公共依赖迁移见上方新记录。

项目`.venv3128`保留RNA专用依赖。当前已启用system-site-packages，torch实际从全局`C:/Program Files/Python312/Lib/site-packages/torch`导入，不再使用项目副本。`run.ps1`和VS Code仍使用项目venv，不依赖旧终端PATH。全局升级共享GPU包会影响使用它的项目，因此本机基础包版本固定在共享清单。

其他项目可以直接选择系统解释器；需要保留项目专用依赖时，在该项目目录用下面命令创建新的共享venv（已有环境先核对，不批量覆盖）：

```powershell
$rnaSystemPython = 'C:\Program Files\Python312\python.exe'
& $rnaSystemPython -m venv --system-site-packages .venv
& '.\.venv\Scripts\python.exe' -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

独立venv默认不会自动看到全局包，只有系统解释器或启用system-site-packages的venv才共享。没有擅自修改其他项目的环境。机器PATH已有Python312与Scripts，安装前已打开的终端/应用需重新打开；旧会话可先使用解释器完整路径。

复现顺序：系统安装3.12.8 → scripts/setup_shared_gpu.ps1（全局基础包与GPU实测）→ scripts/setup_shared_common.ps1（全局公共依赖）→ scripts/setup_environment.ps1（4个RNA专用依赖）。setup_environment不再在.runtime重复安装Python，也不会自动把其他项目迁移到共享环境。

下方为历史环境搭建记录；其中项目本地Python基础安装路径只表示首次搭建状态。

数据清洗阶段新增Gemmi/Barnaba/MDTraj等依赖，见requirements-data.txt；NumPy现固定为2.2.6以兼容Barnaba API。requirements-lock.txt仅保留迁移前历史版本；当前公共版本见requirements-shared-common.txt，RNA专用版本见requirements-data.txt。早期使用的已验证微软签名项目本地DLL（.runtime/vc）保留作兼容回退。

## 系统VC++运行库已补齐（2026-10-27）

按用户明确要求，从[微软官方v14 x64入口](https://aka.ms/vc14/vc_redist.x64.exe)下载并验证Microsoft Corporation有效数字签名后，安装Microsoft Visual C++ Redistributable x64 **14.51.36247.0**。安装退出码0，无需重启；系统注册表Installed=1，System32中的vcruntime140/msvcp140/vcruntime140_1版本均为14.51.36247.0。

全新Python进程直接导入torch、Gemmi、MDTraj和SciPy已不再出现运行库缺失提示；三张RTX 2080 Ti重新通过真实CUDA前向、反向、优化器及Ghost/RWKV四步反向检查。Python3.12.8、PyTorch2.7.1+cu128保持原配置。安装器和日志在本地.runtime/installers，安装审计与环境实测在data/reports；不提交二进制安装包，也不重跑已揭盲模型评估。

该提示来自PyTorch尝试载入三个原生DLL时的预检查。此前项目本地DLL支持实际训练，因此提示没有导致已完成实验失败；本次补齐了系统级依赖。下方首次环境搭建记录中的预检查警告属于历史状态。

## 目标与位置

- 当前系统Python 3.12.8：`C:/Program Files/Python312/python.exe`（原`.runtime/python3128`为历史基础安装）
- 正式虚拟环境：`.venv3128/Scripts/python.exe`
- PyTorch 2.7.1+cu128，CUDA runtime 12.8，适配当前 RTX 2080 Ti（Turing / sm_75）。最终是否正常以 GPU 前向、反向和优化器实测为准。
- 本机发现三张 RTX 2080 Ti，每张 22528 MiB，驱动 576.80。初版模型单卡运行。
- 安装清单：requirements-cuda.txt；历史完整版本：requirements-lock.txt；当前版本见共享清单和requirements-data.txt。

Python 来自本地 `D:/E backup/下载/python-3.12.8-amd64.exe`，安装前验证了 Python Software Foundation 的有效签名。安装为项目专用路径，不添加 PATH、不创建全局启动器。

`D:/stockalarm/requirements.txt` 是参考，其中没有固定 torch/CUDA；没有把股票业务依赖全搬入 RNA 项目。RNA 初版另外安装 numpy/pandas/matplotlib/tqdm/requests/psutil/biopython/pytest 以支持后续数据工作。

早期探查留下的 `.venv` 是 Python 3.12.14 的 CPU 临时环境，**不是正式环境**。所有脚本入口和编辑器配置均明确指向 `.venv3128`。

## CUDA 与 cuDNN

当前使用 PyTorch 官方 CUDA wheel 中的运行时和 cuDNN。普通 PyTorch 训练不要求安装系统级 CUDA Toolkit。只有未来编译自定义 CUDA 扩展才需要匹配的 nvcc/工具链。因此未运行下载目录的 CUDA/驱动/cuDNN 系统安装包，也未覆盖现有驱动。

官方版本安装入口：https://pytorch.org/get-started/previous-versions/ （2.7.1 / cu128）。

## 命令

```powershell
Set-Location 'D:\RNA flod'
./run.ps1 scripts/check_environment.py
./run.ps1 -m pip check
./run.ps1 -m unittest discover -s tests -v
./run.ps1 scripts/train_baseline.py --epochs 5 --batch-size 4 --device cuda:0
```

不用激活环境。也可直接运行 `& '.\.venv3128\Scripts\python.exe' ...`。

`train_baseline.py`默认使用冻结的150k `SequencePairModel` CNN 和`data/experimental/joint_split_v2`；`--model-kind pair_baseline`保留旧小模型回退。训练/评价入口均要求`joint_split_v2_validation.json`通过且train/validation文件SHA256匹配。默认评价仅开放validation；检查点保存模型配置、组宏F1、数据manifest/划分指纹和阈值，评价重新核验这些指纹；`predict.py`按checkpoint配置加载模型并沿用阈值。旧检查点若缺数据指纹，需显式传入`--allow-legacy-checkpoint`才能在validation上作历史诊断。smoke只验证工程链路，不代表模型精度。

复现安装使用 `scripts/setup_environment.ps1`。该脚本不会修改已有的 stockalarm 或旧 Nezha 环境。

环境检查结果：`data/reports/environment.json`。短训练结果：`runs/smoke_gpu/`。短训练只证明数据→训练→保存→评估→预测链路接通，不能作为模型精度或泛化结论。

首次环境搭建实测（历史，当前状态见上方系统VC++安装记录）：3 张卡全部通过基线前向/反向/优化器和 Ghost/RWKV 多步反向；13 项测试通过；pip check 无冲突；GPU 0 完成 32 条样本的 2 epoch 链路验证。当时PyTorch导入曾有VC++ runtime预检查提示，未屏蔽；后续系统运行库安装后提示消失。实际CUDA运算已通过。首次加载 CUDA DLL 明显慢于后续启动。
