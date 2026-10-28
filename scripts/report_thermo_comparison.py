"""Render all frozen conditions and descriptive pair-type strata, without tuning."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from run_thermo_comparison import ROOT, PROTOCOL, checked_protocol, file_sha256, write
from rna2d.data import observable_pairs, pair_metrics
from rna2d.thermodynamics import pair_type


def type_counts(rows, details):
    counts=defaultdict(Counter)
    for row,detail in zip(rows,details):
        if row['id']!=detail['id']:raise ValueError('Row/detail order mismatch')
        truth=observable_pairs(row['pairs'],row['observed_mask'])
        predictions=detail['evaluated_predicted_pairs']
        for kind in ('AU','GC','GU'):
            metrics=pair_metrics([p for p in predictions if pair_type(row['sequence'],p)==kind],
                                 [p for p in truth if pair_type(row['sequence'],p)==kind])
            counts[kind].update({k:metrics[k] for k in ('tp','fp','fn')})
    return {k:dict(v) for k,v in counts.items()}


def main():
    p,datasets=checked_protocol()
    summary_path=ROOT/'data/manifests/thermo_comparison_summary_v1.json'
    summary=json.loads(summary_path.read_text())
    if not summary['all_reload_checks_passed'] or summary['protocol_sha256']!=file_sha256(PROTOCOL):
        raise ValueError('Need completed verified results before rendering')
    baseline_details=json.loads((ROOT/'runs/thermodynamics_v1/step1/validation_seed42_details.json').read_text())['thermo']
    strata={'protocol_sha256':file_sha256(PROTOCOL),'summary_sha256':file_sha256(summary_path),
            'role':'predeclared descriptive strata; no new model/threshold/alpha selection',
            'source_sha256':file_sha256(Path(__file__)),
            'thermo_pair_type':type_counts(datasets['validation'].rows,baseline_details),'conditions':{},'test_accessed':False}
    for condition in p['conditions']:
        records=[]
        for seed in p['seeds']:
            path=ROOT/f'runs/thermodynamics_v1/step3/{condition}/seed{seed}/result.json'
            result=json.loads(path.read_text())
            records.append({'seed':seed,'result_sha256':file_sha256(path),
                            'pair_type_counts':type_counts(datasets['validation'].rows,result['splits']['validation']['details'])})
        strata['conditions'][condition]=records
    write(ROOT/'data/manifests/thermo_comparison_strata_v1.json',strata)
    c=summary['conditions'];b=summary['thermo'];gate=summary['primary_gate']
    lines=['# 热力学修正固定预算对照结果 v1','',
        '第三步完成：2026-10-28（Asia/Taipei）。9次正式训练，每次550更新；9个检查点在独立进程重载后，train/validation逐RNA指标及解码配对均精确一致。未打开旧test。',
        '',f"协议SHA256 `{file_sha256(PROTOCOL)}`；见[冻结协议](THERMO_COMPARISON_PROTOCOL_V1.md)。固定主条件为alpha=.25，alpha=1仅敏感性；没有按结果改alpha/阈值/种子/预算。",'',
        '## 预测与训练成本','',
        '| 方法 | validation组宏F1均值±seed标准差 | seed42 / 43 / 44 | 训练组宏F1均值 | 平均训练秒 | 峰值分配显存均值MiB |',
        '| --- | --- | --- | ---: | ---: | ---: |',
        f"| 热力学配对概率 | {b['group_macro_f1']:.6f}（确定性） | 同一固定预测 | — | 无训练 | BPP计算在CPU |"]
    for name,d in c.items():
        scores=' / '.join(f'{v:.6f}' for v in d['group_macro_f1_seeds'])
        lines.append(f"| {name} | {d['group_macro_f1_mean']:.6f} ± {d['group_macro_f1_stdev']:.6f} | {scores} | {d['train_group_macro_f1_mean']:.6f} | {d['mean_training_seconds']:.3f} | {d['mean_peak_gpu_allocated_MiB']:.3f} |")
    lines += ['', '纯CNN三个种子的最终权重均与历史同预算训练精确一致。150019参数、相同采样和初始化；固定alpha不增加参数。',
        f"评估范围仅为可靠观测端点：逐RNA可观测位置比例平均 {b['metrics']['mean_evaluated_position_fraction']:.2%}；不能把该F1称为完整RNA结构准确率。只有 {b['metrics']['crossing_positive_samples']} 条RNA含可评价交叉真边，热力学方法在这些样本上的交叉边宏F1为 {b['metrics']['crossing_macro_f1_on_positive_samples']:.6f}，并未解决假结。",
        f"先验生成98条序列耗时 {summary['prior_generation_seconds_separate']:.3f} 秒，另外计入；训练计时含缓存读取/拼装。完整先验矩阵是O(N²)特征，不能称其为零显存/零计算。这里速度/显存只描述本配置，不是架构排名。",'',
        '## 主比较及停止规则','',
        f"预声明alpha=.25开发门槛：**{'通过' if gate['passed'] else '未通过'}**。相对热力学基线的平均组宏F1差值为 {gate['mean_gain_vs_thermo']:+.6f}。"]
    for name,passed in gate['rules'].items():lines.append(f"- {name}: {'通过' if passed else '未通过'}")
    lines += ['', 'alpha=1作为完整披露的敏感性结果，不替代主条件或用于从该开发集挑选新默认。即使超过CNN，也只有同时优于强热力学参照才是修正收益候选。主门槛为工程继续标准，非统计显著性证明；三个seed不是三个独立生物学测试集。','',
        '## 逐组结果','',
        '| 隔离组（SHA前12位） | 热力学 | CNN均值 | alpha=.25均值 | alpha=1均值 | .25减热力学 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for group,score in b['per_group'].items():
        lines.append(f"| {group[:12]} | {score:.6f} | {c['cnn']['mean_per_group_f1'][group]:.6f} | {c['hybrid_a025']['mean_per_group_f1'][group]:.6f} | {c['hybrid_a1']['mean_per_group_f1'][group]:.6f} | {gate['per_group_delta_vs_thermo'][group]:+.6f} |")
    lines += ['', '这些是隔离/同源组，不是已核实Rfam家族。逐RNA可观测比例、TP/FP/FN、F1和完整/掩码后配对列表在本地 runs/thermodynamics_v1/step3/<condition>/seed*/result.json；未将逐RNA审计明细发布成训练集。','',
        '## 长度分层（逐RNA宏F1，seed均值）','',
        '| 长度nt | RNA数 | 热力学 | CNN | alpha=.25 | alpha=1 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for length,d in b['metrics']['length_bins'].items():
        lines.append(f"| {length} | {d['samples']} | {d['macro_f1']:.6f} | {c['cnn']['length_strata'][length]['mean_macro_f1']:.6f} | {c['hybrid_a025']['length_strata'][length]['mean_macro_f1']:.6f} | {c['hybrid_a1']['length_strata'][length]['mean_macro_f1']:.6f} |")
    lines += ['', '长129–256 nt仅1条，分层差异不能推断长度效应；更长验证样本缺失。','',
        '## 配对类型分层（平均TP / FP / FN）','',
        '| 类型 | 热力学 | CNN | alpha=.25 | alpha=1 |','| --- | --- | --- | --- | --- |']
    for kind in ('AU','GC','GU'):
        z=strata['thermo_pair_type'][kind];cells=[' / '.join(str(z[k]) for k in ('tp','fp','fn'))]
        for condition in p['conditions']:
            cells.append(' / '.join(f"{statistics.mean(x['pair_type_counts'][kind][k] for x in strata['conditions'][condition]):.2f}" for k in ('tp','fp','fn')))
        lines.append(f"| {kind} | "+' | '.join(cells)+' |')
    lines += ['', '分层仅覆盖AU/GC/GU及当前可观测端点，不能外推到非经典配对或完整假结集合。独有真边/假阳性和双方漏边逐seed汇总在summary。','',
        '## 可以与不可以得出的结论','',
        f"冻结决策：{summary['decision']}。",'',
        '本实验区分“利用强先验后超过小样本CNN”和“神经网络真正改善热力学预测”。只支持当前88训练/10开发RNA及固定计算条件；没有新增独立测试，不能证明跨家族泛化。先验不包含实验离子/配体/复合物环境，预测概率也未经实验校准。所有观察都不等于真实折叠过程或构象系综验证。',
        '', '不改生产默认CNN、不接回J/Ghost/RWKV、不扩大同一开发集上的调参。后续应保留强热力学参照，先获得新的独立实验评估组，或在有明确条件证据的错误类型上另立研究问题；不以增加当前CNN训练时长作为已证明的解决方案。',
        '', '机器结果：data/manifests/thermo_comparison_summary_v1.json、thermo_comparison_strata_v1.json。',
        '', '工程验证：本步结束后全套157项测试通过。核验入口最初使用Python -I，因隔离模式不包含脚本目录而未能导入同目录runner；改用普通venv启动命令后完成全部独立重载，冻结训练/核验源码未更改。报告渲染的路径类型错误已修复，不涉及训练重试或指标变更。',
        '', '复核命令（已有运行，不能重复训练覆盖）：','', '```powershell',
        './run.ps1 scripts/verify_thermo_comparison.py', './run.ps1 scripts/report_thermo_comparison.py', '```']
    (ROOT/'docs/THERMO_COMPARISON_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(summary['decision'])


if __name__=='__main__':main()
