"""Report every frozen seed, control and horizon, without selecting winners."""
import json
import statistics
from acquire_experimental_corpus import ROOT,write_json,digest_file


def main():
    reports=[json.loads((ROOT/'data/manifests'/f'gate3_seed{s}_v1.json').read_text()) for s in (42,43,44)]
    protocol=json.loads((ROOT/'data/manifests/gate3_protocol_v1.json').read_text())
    assert all(r['protocol_sha256']==digest_file(ROOT/'data/manifests/gate3_protocol_v1.json') for r in reports)
    aggregate=[];diagnostics=[];effects=[]
    for variant in protocol['variants']:
        entries=[next(e for e in r['results'] if e['variant']==variant) for r in reports]
        assert all(e['reload_verified'] for e in entries)
        for hi,h in enumerate(protocol['horizons']):
            rows=[e['validation']['ordered'][hi] for e in entries]
            values=[r['standardized_dynamic_mse'] for r in rows]
            linear=[r['baselines']['linear'][hi]['standardized_dynamic_mse'] for r in reports]
            mean_change=[r['baselines']['train_mean_change'][hi]['standardized_dynamic_mse'] for r in reports]
            aggregate.append(dict(variant=variant,horizon=h,mse_mean=statistics.mean(values),mse_seed_std=statistics.pstdev(values),skill_persistence_mean=statistics.mean(r['skill_vs_persistence'] for r in rows),skill_linear_mean=statistics.mean(1-a/b for a,b in zip(values,linear)),skill_train_mean_change=statistics.mean(1-a/b for a,b in zip(values,mean_change)),mse_by_channel_mean=[statistics.mean(r['mse_by_channel'][c] for r in rows) for c in range(5)],sign_accuracy_dynamic_mean=[statistics.mean(r['sign_accuracy_by_dynamic_channel'][c] for r in rows if r['sign_accuracy_by_dynamic_channel'][c] is not None) for c in range(3)]))
            effects.append(dict(variant=variant,horizon=h,skill_vs_zero_effect_mean=statistics.mean(e['action_effect'][hi]['skill_vs_zero_action_effect'] for e in entries)))
            for mode in entries[0]['validation']:
                metrics=[e['validation'][mode][hi]['standardized_dynamic_mse'] for e in entries]
                diagnostics.append(dict(variant=variant,horizon=h,mode=mode,mse_mean=statistics.mean(metrics),mse_seed_std=statistics.pstdev(metrics)))
    comparisons=[]
    for h in protocol['horizons']:
        for control in ('ghost_obs','ghost_momentum','ghost_action','ghost_no_rwkv_memory','mlp_full'):
            deltas=[];group_deltas={}
            for report in reports:
                hi=protocol['horizons'].index(h)
                full=next(e for e in report['results'] if e['variant']=='ghost_full')['validation']['ordered'][hi]
                other=next(e for e in report['results'] if e['variant']==control)['validation']['ordered'][hi]
                deltas.append(full['standardized_dynamic_mse']-other['standardized_dynamic_mse'])
                om={r['group']:r['standardized_dynamic_mse'] for r in other['per_group']}
                for row in full['per_group']:group_deltas.setdefault(row['group'],[]).append(row['standardized_dynamic_mse']-om[row['group']])
            comparisons.append(dict(horizon=h,control=control,full_minus_control_mse_mean=statistics.mean(deltas),full_better_seeds=sum(d<0 for d in deltas),full_better_groups_after_seed_average=sum(statistics.mean(ds)<0 for ds in group_deltas.values()),groups=len(group_deltas)))
    summary=dict(aggregate=aggregate,diagnostics=diagnostics,action_effect=effects,paired_comparisons=comparisons,new_independent_test=False,all_checkpoint_reloads_passed=True,result_sha256={str(s):digest_file(ROOT/'data/manifests'/f'gate3_seed{s}_v1.json') for s in (42,43,44)})
    write_json(ROOT/'data/manifests/gate3_summary_v1.json',summary)
    lines=['# 第三关：全部因果诊断结果','','三种子，固定500更新；35条训练序列、5条开发验证序列。模拟辅助转移，不是实验轨迹或序列从头折叠。全部配置与未来间隔均报告。','', '| 模型 | 未来步数 | 标准化动态MSE ±种子标准差 | Skill对持久性 | Skill对线性 |','| --- | --- | --- | --- | --- |']
    for r in aggregate:lines.append(f"| {r['variant']} | {r['horizon']} | {r['mse_mean']:.6g} ± {r['mse_seed_std']:.3g} | {r['skill_persistence_mean']:.4f} | {r['skill_linear_mean']:.4f} |")
    lines+=['','## 完整模型信息干预','','| 未来步数 | 输入方式 | 标准化动态MSE |','| --- | --- | --- |']
    for r in diagnostics:
        if r['variant']=='ghost_full':lines.append(f"| {r['horizon']} | {r['mode']} | {r['mse_mean']:.6g} |")
    lines+=['','## ±动作效应预测','','相同起点的正负动作预测差与实际模拟差比较，Skill=0对应完全不预测动作效应。','', '| 模型 | 未来步数 | 动作效应Skill |','| --- | --- | --- |']
    for r in effects:lines.append(f"| {r['variant']} | {r['horizon']} | {r['skill_vs_zero_effect_mean']:.4f} |")
    lines+=['','## 完整模型与独立控制的配对比较','','负差值表示完整模型更好；种子数3、验证组数5，胜出次数不是显著性检验。','', '| 未来步数 | 对照 | 完整模型减对照MSE | 完整模型较好种子数 | 较好组数 |','| --- | --- | --- | --- | --- |']
    for r in comparisons:lines.append(f"| {r['horizon']} | {r['control']} | {r['full_minus_control_mse_mean']:.6g} | {r['full_better_seeds']}/3 | {r['full_better_groups_after_seed_average']}/{r['groups']} |")
    lines+=['','逐通道误差、变化方向、每个组和种子、训练误差、参数及检查点SHA见公开JSON。完整条件的动量仅为摘要，不是完整相空间。历史打乱、drop/reset是干预诊断，不能单独作为架构因果证据；应结合独立训练控制。原总线等未使用参数使参数量匹配不等于有效容量/FLOPs匹配。','', '与旧物理预测实验同时改变了数据分叉、未来间隔、残差目标和缩放，不能用跨版本差异单独证明残差形式的因果收益。没有新独立盲测，不据本表宣布普遍RNA折叠能力。','']
    (ROOT/'docs/GATE3_RESULTS_V1.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(dict(h16=[r for r in aggregate if r['horizon']==16],h16_comparisons=[r for r in comparisons if r['horizon']==16]),indent=2))


if __name__=='__main__':main()
