# 项目工作约定

- 本目录是新项目；不要修改 D:/Nezha/Nezha-bio。
- 主线：1D 序列 → 2D 配对预测；保留 Ghost、RWKV、分步更新作为逐项验证的研究机制。
- 先建立数据来源、划分、评估和简单基线，再增加模块。未经消融验证不得宣称模块有效。
- 计算迭代步不是物理时间；不能用最终结构的前缀裁剪冒充共转录中间态标签。
- 原始数据不可覆盖；记录 URL、下载时间、SHA256、标签来源和清洗排除原因。
- unknown 不等于 0。探测反应性不等于配对标签。测试集不得用于选择超参数。
- 完整扩充池不得直接用于训练。joint_split_v2中已核验同源/研究/构建体隔离的88/10/8子集可按冻结实验协议用于探索；其余1D构建体继续隔离，禁止预训练泄漏。小样本不能宣称广泛跨家族泛化。旧61条版本仅作历史链路验证。
- 旧 Ghost/RWKV 代码位于 legacy_reference，只供参考，不直接导入旧 Config。
- 不引入大规模数据下载、3D 支线或额外依赖，除非任务需要；所有实验记录数据指纹、配置、种子、指标和限制。
- 用户已授权恢复原物理状态/观测/反馈接口、接入隔离后的3D并分别验证训练预算/展开长度。three_d_isolated_v1仅35条训练序列/5条开发验证，无新测试；physical_forecast_v1转移是模拟辅助数据，不得混入实验金标准或称为真实折叠轨迹。仍不使用旧已揭盲测试调参。
- 用户明确暂不接回能量驱动J门控。representation_v2仅做静态密度1/2保留/移除与实时几何接触18/19增量的输入析因实验；原目标、J、物理和划分不变。基础观测静态索引为1/2/12/13，本轮仅消融1/2，不将无收益解释为所有静态信息无用。
- 金标准池仅用直接实验证据，排除模型预测/补全标签与低质量数据。允许对可靠比较序列分析建立独立辅助池，但须逐条核验文献、比对映射和验证方式，不能将其当作实验真值。遵循 data/experimental/POLICY.json；旧 data/processed 基准不能自动视为已批准训练数据。

- 用户已授权定义下一轮study_v2数据角色，见docs/STUDY_V2_DATA_ROLES_V1.md与data/manifests/study_v2_role_policy_v1.json。仅在此新研究中，退役历史评估关联不再单独永久否决训练候选；旧成绩不再承担独立验证。当前10条开发及未来新测试的精确/构建体/family-clan/同源/原研究关系仍受保护，未知不当无关联，来源/质量/映射/标签门禁不放宽。角色清单不是训练准入，原数据/旧协议/生产CLI不变；新数据与A/B/C须另立准入及冻结协议，不能复用已揭盲旧测试证明新模型效果。

- 本机基础环境现为系统Python3.12.8（C:/Program Files/Python312/python.exe）与全局torch2.7.1+cu128/CUDA runtime12.8/cuDNN9.7.1。RNA的.venv3128启用system-site-packages并共享全局torch及公共依赖，仅保留barnaba/biopython/gemmi/mdtraj四个RNA专用包；run.ps1仍调用venv。不要重装.runtime/python3128、在venv重复安装torch、擅自升级全局GPU版本或修改其他项目venv；共享配置与复现见docs/ENVIRONMENT.md、requirements-shared-gpu.txt与requirements-shared-common.txt；公共包冲突按D:/stockalarm/requirements.txt固定版本，不再按历史requirements-lock.txt覆盖全局。
