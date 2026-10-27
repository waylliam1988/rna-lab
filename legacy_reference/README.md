# 旧实现参考快照

2026-10-27 从 D:/Nezha/Nezha-bio 复制 ghost.py、kernels.py、config.py。
保留原文件，不在新训练主线导入。迁移时先解除全局 Config 耦合，再定义显式状态接口、样本间重置和可验证预测目标。

这里并不是已经完成接入的 Ghost/RWKV 新模型。文件校验和见 snapshot_manifest.json。
