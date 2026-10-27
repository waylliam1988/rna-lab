"""Render frozen public experiment summaries; never execute model selection."""
import json
from acquire_experimental_corpus import ROOT


def main():
    read=lambda p:json.loads((ROOT/p).read_text(encoding='utf-8'))
    report=read('data/manifests/final_evaluation_v1.json')
    plan=read('data/manifests/final_evaluation_runs_v1.json')
    lines=['# 固定协议实验结果 v1（2026-10-27）','',
           '176条配对标签中106条通过本轮同源/研究/构建体证据隔离：88/10/8条训练/验证/测试，共8/7/4个声明组。未核验1D构建体没有进入预训练。',
           '', '全部配置各3种子、25epoch，以验证宏F1选checkpoint。所有方案结束后，按预声明的10种配置统一揭盲测试一次；测试仅8条、4组。数值为种子均值±标准差，不是家族总体置信区间。','',
           '| 配置 | 参数 | 验证宏F1 | 测试宏F1 | 每RNA推理毫秒 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for name in plan['runs']:
        s=read('data/manifests/'+name+'.json');r=report['results'][name]
        lines.append(f"| {name} | {s['seeds'][0]['parameters']} | {s['validation_macro_f1_mean']:.4f} ± {s['validation_macro_f1_stdev']:.4f} | {r['test_macro_f1_mean']:.4f} ± {r['test_macro_f1_stdev']:.4f} | {1000*r['mean_inference_seconds_per_RNA']:.3f} |")
    lines.extend(['', '## 判断与限制','',
        '本轮封存测试没有支持普通迭代、RWKV、Ghost或完整组合比静态模型有稳定优势。不能从如此小的测试集推断这些方法一般无效，也不能反过来用验证集较高的均值声称突破。', '',
        'RWKV、Ghost-only和完整组合的算法下一特征预测均未胜过持久性基线，没有获得世界模型验证。目标是模型自身的算法状态，不是实验共转录中间态或真实RNA动力学。','',
        'Ghost-only只含迁移核心的两级连续控制和轨迹编码，与原Ghost进入记忆前的数值已对齐；完整组合额外接入一次RWKV。每RNA/残基的控制状态显式传递，每次forward重新初始化，跨位点经配对上下文通信，不跨样本携带。50项测试通过，包括数值等价、零驱动、padding、状态重置、辅助目标梯度和有限性。','',
        '普通迭代的1/2/4/8步诊断只在验证集进行。更新幅度下降不等于F1单调改善；最终固定4步，没有逐目标挑最佳步。按组、上下文、长度、交叉配对指标、每种子训练时长和显存见完整机器摘要与本地日志。','',
        '同epoch/样本预算，静态宽度按参数量选择，参数近似匹配；没有精确匹配训练FLOPs。推理时间包含诊断、搬运、CPU解码和评估开销，不能当作优化内核吞吐。额外容量、计算、辅助损失和样本量限制因果归因。','',
        '## 文件与复核','',
        '- 逐项执行与提交顺序见 [STAGED_VALIDATION.md](STAGED_VALIDATION.md)。',
        '- 冻结协议与预声明配置：`data/manifests/experiment_protocol_v1.json`、`mechanism_controls_v1.json`、`final_evaluation_runs_v1.json`。',
        '- 全部测试种子、checkpoint哈希、组均值、上下文与效率：`data/manifests/final_evaluation_v1.json`。',
        '- 权重和训练日志：`runs/stage3_*`至`runs/stage6_*`；逐目标测试预测：`runs/final_evaluation_v1/`。原始数据/权重仅留本地。','',
        '## 接下来','',
        '优先补齐构建体母本/接头与文献同源证据、独立注释缺失/定义分歧，扩充独立家族监督和评估覆盖。当前不升级3D或继续增加模型复杂度。新假设在训练/验证上开发；本轮测试已经揭盲，后续改动需要新增未见目标或独立预先冻结的评估，不能反复对这8条刷分。'])
    (ROOT/'docs/EXPERIMENT_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
