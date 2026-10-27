# 核心迁移范围

## 已迁入 src/rna2d/core

- `RWKVCellLite`：从旧 kernels.py 按 AST 提取，算法正文逐字保留；独立 imports，无旧 Config。
- `ContinuousControlLayer`：从旧 ghost.py 提取，保留膜状态、辅助迹、双 tanh 归一化和 running_activity 诊断统计。
- `GhostDynamics`：重写旧 NeuralCortex 的有效主路径，保留双阶段连续动力学、轨迹编码、零基线扣除、RWKV 记忆和有界控制率。
- `GhostSettings`：显式小配置，替代旧全局 Config。

GhostDynamics 的样本状态由调用方传入/返回；`state=None` 明确重置。模块内只保留训练诊断的 running_activity，不缓存用于下一样本预测的隐状态。默认保留完整梯度，截断必须显式调用 detach_state。

prediction_head 是未定义监督目标的通用预测头；默认 5 维用于核对旧头的数值迁移，不再声称代表 Rg/力/能量，也不能仅凭这个头称为世界模型。后续按 2D 状态设计目标后修改 prediction_dim 并训练。

## 暂未迁入主线

- 旧 Ghost 外层 USBLink/ProtocolSpec 总线、目标投影、旧检查点兼容逻辑。
- BioSystem、3D 力场、Flow/GNO、旧训练器、复杂损失与课程配置。
- 旧权重：没有隐式加载；只保留代码参考和 SHA256。

这些不代表被永久放弃。先验证独立动力学和显式状态，再决定总线等机制是否必要。

## 验证

tests/test_core.py 检查：RWKV/控制原语提取一致性、旧 Cortex 与新核心在相同权重/输入/状态下的输出一致性、逐步状态、重置隔离、长展开有限值及梯度。旧 Cortex 仅通过测试隔离执行类定义，不导入旧项目。

这些测试通过只证明迁移和数值接口正确，不证明 RNA 预测收益。现已完成普通迭代、RWKV、Ghost及组合的2D适配实验；它们是新建的配对矩阵更新器，不等于原Nezha完整物理折叠系统。完整迁移边界和3D预审见 [NEZHA_FIDELITY_AND_3D_PLAN.md](NEZHA_FIDELITY_AND_3D_PLAN.md)。
