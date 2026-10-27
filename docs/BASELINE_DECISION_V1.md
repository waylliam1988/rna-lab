# 当前2D工程基线：继续使用CNN

决策形成2026-10-27，执行状态同步2026-10-28。当前生产入口已采用此CNN；本次文档同步不新增训练、不改变冻结协议。

当前主任务是sequence-only的1D序列→带观测掩码的2D配对预测。继续使用已验证的CNN作为工程基线，保留RWKV、Ghost和分步更新研究实现；不把这项选择解释为CNN架构的普遍优势。

## 选择依据

- 能够充分拟合：约150k参数的比较版CNN在88条训练RNA上，三个种子最终宏F1为0.9902、0.9723、0.9863；这是优化可用性的证据。
- 工程成本低：同一GPU、float32、batch1、256nt下，网络推理中位数约2.28ms，含CPU解码约13.59ms，训练更新约9.81ms，训练峰值allocated约74.5MiB。该测量含当前进程库工作区，不是显卡总占用，也不是22GB下的最大长度保证。
- 比较边界清楚：三个其他编码器在三个lr、一个固定warmup、25epoch和三个种子下，最终训练F1全部低于0.8，已止损。因此旧验证分数不能证明CNN泛化更好；目前它是唯一达到充分拟合门槛的受检实现。
- 容易固定对照：保持编码器、配对头、掩码、解码及预算不变，之后可单独检验数据扩充或研究机制的增量收益。任何机制收益仍需独立消融和新评估证据。

## 仍然存在的问题

CNN原比较开发宏F1只有0.4332±0.0419（10条、7组），与训练F1相差很大。能拟合不等于能泛化，不能称为高精度折叠预测器。当前CNN编码器感受野为17；共享配对头可对任意距离的位置打分，但不意味着编码器已学会全局折叠。输出配对矩阵仍有N×N开销。

当前标签是保守AU/GC/GU、单伙伴、观测端点范围内的配对监督，不是完整非经典相互作用图。贪心解码允许交叉边，不证明假结已被解决。RWKV-Lite不是完整官方RWKV；EGNN-seed只有人工序列位置坐标，不能据此否定真实3D几何网络。

## 避免基线版本混淆

| 实现 | 用途 | 记录 |
| --- | --- | --- |
| `src/rna2d/model.py`的PairBaseline | 早期链路和stage3–6小参数对照 | [早期实验](EXPERIMENT_RESULTS_V1.md) |
| `src/rna2d/sequence_encoders.py`的SequencePairModel(kind=cnn,width=113) | 本次约150k参数比较与训练拟合参考，hidden64/depth2 | [比较结果](ENCODER_COMPARISON_RESULTS_V1.md)、[拟合扫描](ENCODER_LR_SCAN_RESULTS_V1.md) |

训练、评价和推理入口现默认使用第二种`SequencePairModel(kind=cnn,width=113,hidden=64,depth=2)`，并从`joint_split_v2`加载数据；训练checkpoint记录明确模型配置、数据manifest/划分指纹和评价阈值。`PairBaseline`仍可通过`train_baseline.py --model-kind pair_baseline`显式回退，模型工厂继续加载其历史checkpoint。该入口和新smoke链路不构成新性能证据，也不改写上面的旧实验结果。

## 后续已完成的损失对照

2026-10-28的S6-L v2使用同一CNN、组均匀采样、每条件三个种子550更新，BCE验证逐RNA宏F1为0.4710、组宏F1为0.5224；加伙伴竞争组宏F1为0.5248，未过预声明gate，保持BCE默认。该结果不替换上方原比较的0.4332：采样及报告口径不同，也没有新独立测试。见[损失报告](DATA_FIRST_LOSS_RESULTS_V2.md)。

## 下一步

固定这版CNN及现有结果作为工程参照。生产训练脚本按validation group-macro F1保存开发checkpoint；一般CLI训练并不替代25epoch/550-update冻结实验runner。评价CLI默认只开放validation且校验checkpoint与数据manifest指纹；独立test仍须另立冻结评估协议。优先扩充经过原研究/同源/构建体审核的独立监督和新冻结评估，报告长度、组别和标签覆盖；已揭盲的8条旧测试不再用于调参。未通过隔离的1D池不进入预训练。本轮编码器调优已结束；暂不重开验证排名或接回J门控。

证据：[EGNN几何审计](EGNN_GEOMETRY_AUDIT_V1.md)、[有限lr扫描](ENCODER_LR_SCAN_RESULTS_V1.md)、[工程性能](ENCODER_COMPARISON_RESULTS_V1.md)。
