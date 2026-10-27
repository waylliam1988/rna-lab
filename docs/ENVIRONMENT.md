# 本地环境

数据清洗阶段新增Gemmi/Barnaba/MDTraj等依赖，见requirements-data.txt；NumPy现固定为2.2.6以兼容Barnaba API。Windows原生运行时采用验证微软签名的项目本地DLL（.runtime/vc），可通过scripts/prepare_native_runtime.ps1复现，不修改系统文件。完整已安装版本以requirements-lock.txt为准。

## 目标与位置

- Python 3.12.8：`.runtime/python3128/python.exe`
- 正式虚拟环境：`.venv3128/Scripts/python.exe`
- PyTorch 2.7.1+cu128，CUDA runtime 12.8，适配当前 RTX 2080 Ti（Turing / sm_75）。最终是否正常以 GPU 前向、反向和优化器实测为准。
- 本机发现三张 RTX 2080 Ti，每张 22528 MiB，驱动 576.80。初版模型单卡运行。
- 安装清单：requirements-cuda.txt；实际完整版本：requirements-lock.txt（安装完成后生成）。

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

复现安装使用 `scripts/setup_environment.ps1`。该脚本不会修改已有的 stockalarm 或旧 Nezha 环境。

环境检查结果：`data/reports/environment.json`。短训练结果：`runs/smoke_gpu/`。短训练只证明数据→训练→保存→评估→预测链路接通，不能作为模型精度或泛化结论。

本轮实测：3 张卡全部通过基线前向/反向/优化器和 Ghost/RWKV 多步反向；13 项测试通过；pip check 无冲突；GPU 0 完成 32 条样本的 2 epoch 链路验证。PyTorch 导入仍有 VC++ runtime 预检查提示，未屏蔽；实际 CUDA 运算已通过。首次加载 CUDA DLL 明显慢于后续启动。
