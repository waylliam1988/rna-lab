# 原物理接口恢复与验证

2026-10-27，阶段一。旧项目只读；恢复脚本scripts/restore_nezha_physics.py按AST选择定义，输出到src/rna2d/physics。来源文件SHA256、53项以上实际配置及逐文件输出哈希记录在data/manifests/nezha_physics_restoration_v1.json。

恢复范围：完整Ghost/NeuralCortex/USBLink/ProtocolSpec，原RWKV与连续控制类，解析RNA力场、几何不变量、方向投影，Nose–Hoover/噪声积分器，原宏观观测及下一状态smooth-L1反馈。原Ghost配置为256/512/8，默认温度0、dt0.05、gamma0.35。未加载旧检查点。

PhysicalInterface是新写的显式适配层：q[B,L,6]前三维nm、后三维单位方向，p为广义动量，xi为温控状态；接口接收外部port_force，重算解析场、调用原积分器、投影方向，然后重新观测。温度/时间/能量是约化模拟量，未标定为K/s/kcal。Å输入必须先乘0.1，方向不缩放。

原观测前五维为Rg、输入J的平均/最大行密度、六维力RMS、按残基归一化并asinh压缩的能量；J固定时两项接触观测是常量，不能用它们证明动力学预测。可追加原residue_invariants的均值。下一状态目标对Rg/力RMS做log1p，其余保持原变换；标签detach，预测保留梯度。

tests/test_physics_contract.py验证六项：原定义/方法AST一致；解析位置力与能量负梯度、方向力与切向负梯度一致；平移不变量/padding/反馈零误差；零温重放及动作改变下一状态；原尺寸Ghost多步反传；噪声随温度及逆质量平方根缩放。它们是接口、局部数学与数值验证，不是实验折叠准确率证明。

边界：没有恢复完整BioSystem编码器、GVP感知、策略/安全门、Flow/GNO/自适应质量或所有课程/损失组合。适配层明确使用单位逆质量；port_force由调用者提供，不冒充原策略。原积分器返回的H/V不包含完整RNA解析势能，不能作为总RNA能量守恒的证据；实验使用单独compute_rna_force_field_fast的能量。原温控K目标和噪声温度的自由度约定仍需物理标定，不宣称已验证长期平衡分布。

下一阶段只接入经过构建体、同源和原研究联合隔离的静态3D。模拟状态与实验坐标分别记录；不把静态最终结构或扰动重放称为实验轨迹。
