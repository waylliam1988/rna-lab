"""Combine all encoder quality and resource measurements, no model selection."""
import json
import statistics as st
from acquire_experimental_corpus import ROOT,write_json,digest_file


def main():
    p=json.loads((ROOT/'data/manifests/encoder_comparison_protocol_v1.json').read_text());reports=[json.loads((ROOT/'data/manifests'/f'encoder_comparison_seed{s}_v1.json').read_text()) for s in p['seeds']]
    resource_path=ROOT/'data/manifests/encoder_resource_benchmark_isolated_v2.json'
    if not resource_path.exists():resource_path=ROOT/'data/manifests/encoder_resource_benchmark_v1.json'
    resources=json.loads(resource_path.read_text());result=[]
    for kind in p['encoders']:
        entries=[next(r for r in report['results'] if r['encoder']==kind) for report in reports];metrics={}
        for metric in ('macro_f1','macro_precision','macro_recall','micro_f1','group_macro_f1'):
            values=[e['validation'][metric] for e in entries];metrics[metric]=dict(mean=st.mean(values),seed_std=st.pstdev(values))
        result.append(dict(encoder=kind,parameters=entries[0]['parameters'],quality=metrics,all_reloads_verified=all(e['reload_verified'] for e in entries),training_seconds=[e['training_including_logged_validation_seconds'] for e in entries],observed_training_peak_MiB=[e['training_peak_allocated_MiB'] for e in entries]))
    summary=dict(encoders=result,resource_file=resource_path.name,resource_sha256=digest_file(resource_path),prediction_result_sha256={str(s):digest_file(ROOT/'data/manifests'/f'encoder_comparison_seed{s}_v1.json') for s in p['seeds']},test_evaluated=False)
    write_json(ROOT/'data/manifests/encoder_comparison_summary_v1.json',summary)
    lines=['# 序列→2D：编码器预测、速度与显存比较','','**解释更正：各编码器训练拟合程度不同，以下旧验证分数不能证明CNN泛化更好或架构优劣。见[EGNN几何审计](EGNN_GEOMETRY_AUDIT_V1.md)与[训练拟合扫描](ENCODER_LR_SCAN_RESULTS_V1.md)。速度/显存仅作为当前实现的工程测量保留。**','','固定最后epoch25，三种子均报告；88训练/10开发验证、7个验证组，不读取旧测试。参数约150k、共享配对头，结果只对应本轮具体实现。','', '| 编码器 | 参数 | 逐RNA宏F1 ±种子标准差 | 按组宏F1 ±种子标准差 | Precision | Recall |','| --- | --- | --- | --- | --- | --- |']
    for r in result:
        q=r['quality'];lines.append(f"| {r['encoder']} | {r['parameters']} | {q['macro_f1']['mean']:.4f} ± {q['macro_f1']['seed_std']:.4f} | {q['group_macro_f1']['mean']:.4f} ± {q['group_macro_f1']['seed_std']:.4f} | {q['macro_precision']['mean']:.4f} | {q['macro_recall']['mean']:.4f} |")
    lines+=['','## 同GPU性能与显存','','RTX2080Ti，float32，batch1；各5次预热、20次同步计时，中位数。正式表采用每个编码器/长度/模式独立进程，消除前一案例CUDA库持久工作区的继承偏差；初版同进程数据保留供审计，不作为正式显存排序。峰值allocated含模型/输入/当前进程库工作区；训练还含梯度/AdamW，reserved另存JSON；不等于nvidia-smi的进程总显存。合成输入仅用于性能。','', '| 编码器 | 长度 | 网络推理ms | 含CPU解码ms | 推理峰值MiB | 训练更新ms | 训练峰值MiB |','| --- | --- | --- | --- | --- | --- | --- |']
    for kind in p['encoders']:
        for n in p['speed_benchmark']['lengths']:
            inf=next(r for r in resources['cases'] if r['encoder']==kind and r['length']==n and r['mode']=='inference');train=next(r for r in resources['cases'] if r['encoder']==kind and r['length']==n and r['mode']=='training')
            if inf['status']!='ok' or train['status']!='ok':lines.append(f'| {kind} | {n} | OOM/见JSON | — | — | — | — |');continue
            lines.append(f"| {kind} | {n} | {inf['network']['median_ms']:.3f} | {inf['network_plus_cpu_decode']['median_ms']:.3f} | {inf['peak_allocated_MiB']:.1f} | {train['step']['median_ms']:.3f} | {train['peak_allocated_MiB']:.1f} |")
    lines+=['','完整训练耗时含曲线记录时的验证，且种子曾在不同GPU并行，只作工程记录；公平速度排序以上述独占本轮任务的同GPU合成形状测量为准。外部无关系统负载未完全控制。','', 'BiRWKV-Lite是原Nezha轻量cell的双向序列版本，有chunk32前缀优化及有界decay，不是完整官方RWKV或专用融合内核。EGNN-seed使用固定序列位置坐标、稠密边和径向读出，不使用实验几何，不代表3D精修排名。CNN/Transformer同样为本项目小型定制实现。各编码器上下文/FLOPs不同；共享N×N配对头也使RWKV全模型不具纯线性显存。','', '验证只有10条且用于开发，旧测试已揭盲而未复用；本轮不足以给架构家族、长链极限或大规模预训练能力排名。原比较阶段未单独调参；后续训练内有限lr扫描已止损，不替换这里的原验证数值，见ENCODER_LR_SCAN_RESULTS_V1.md。所有检查点重载、padding/对称性/梯度、RWKV递归等价及EGNN层等变性已检查。逐样本/长度/交叉配对和所有种子见公开JSON。','']
    (ROOT/'docs/ENCODER_COMPARISON_RESULTS_V1.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
