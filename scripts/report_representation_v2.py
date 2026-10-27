"""Paired feature effects, including all horizons and five development groups."""
import argparse
import json
import statistics as st
from acquire_experimental_corpus import ROOT,write_json,digest_file


def primary(entry):return st.mean(r['standardized_dynamic_mse'] for r in entry['validation'])


def group_primary(entry):
    values={}
    for row in entry['validation']:
        for group in row['per_group']:values.setdefault(group['group'],[]).append(group['standardized_dynamic_mse'])
    return {g:st.mean(v) for g,v in values.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['static','geometry'],required=True);args=parser.parse_args()
    phases=['static'] if args.phase=='static' else ['static','geometry'];reports={}
    for phase in phases:
        reports[phase]=[json.loads((ROOT/'data/manifests'/f'representation_v2_{phase}_seed{s}.json').read_text()) for s in (42,43,44)]
    def entry(phase,seed,kind,profile):return next(e for e in reports[phase][seed]['results'] if e['kind']==kind and e['profile']==profile)
    rows=[]
    for phase in phases:
        profiles=sorted({e['profile'] for e in reports[phase][0]['results']})
        for kind in ('ghost','mlp'):
            for profile in profiles:
                entries=[entry(phase,s,kind,profile) for s in range(3)];values=[primary(e) for e in entries]
                assert len({e['initial_weights_sha256'] for e in entries})==3
                horizon=[]
                for hi,h in enumerate((1,4,8,16)):
                    x=[e['validation'][hi]['standardized_dynamic_mse'] for e in entries]
                    baseline=[r['baselines']['linear'][hi]['standardized_dynamic_mse'] for r in reports[phase]]
                    horizon.append(dict(horizon=h,mse_mean=st.mean(x),mse_std=st.pstdev(x),skill_vs_linear_mean=st.mean(1-a/b for a,b in zip(x,baseline))))
                rows.append(dict(kind=kind,profile=profile,primary_mse_mean=st.mean(values),primary_mse_seed_std=st.pstdev(values),horizons=horizon))
    comparisons=[]
    validation_meta=json.loads((ROOT/'data/simulated/gate3_v1/validation_metadata.json').read_text())
    group_ids={}
    for row in validation_meta:group_ids.setdefault(row['split_group_id'],set()).add(row['id'])
    pairs=[('drop_static_without_geometry','static','keep_static_no_geom','static','drop_static_no_geom')]
    if args.phase=='geometry':pairs += [('add_geometry_static_kept','static','keep_static_no_geom','geometry','keep_static_geom'),('add_geometry_static_dropped','static','drop_static_no_geom','geometry','drop_static_geom'),('drop_static_with_geometry','geometry','keep_static_geom','geometry','drop_static_geom')]
    for name,base_phase,base,new_phase,new in pairs:
        for kind in ('ghost','mlp'):
            before=[entry(base_phase,s,kind,base) for s in range(3)];after=[entry(new_phase,s,kind,new) for s in range(3)]
            assert all(a['initial_weights_sha256']==b['initial_weights_sha256'] for a,b in zip(before,after))
            deltas=[primary(b)-primary(a) for a,b in zip(before,after)]
            gains=[1-primary(b)/primary(a) for a,b in zip(before,after)]
            group_deltas={}
            for a,b in zip(before,after):
                x,y=group_primary(a),group_primary(b)
                for g in x:group_deltas.setdefault(g,[]).append(y[g]-x[g])
            better=sum(v<0 for v in deltas);better_groups=sum(st.mean(v)<0 for v in group_deltas.values())
            reverse=sum(v>0 for v in deltas);reverse_groups=sum(st.mean(v)>0 for v in group_deltas.values())
            group_effects=[]
            for group in sorted(group_deltas):
                x=st.mean(group_primary(e)[group] for e in before);y=st.mean(group_primary(e)[group] for e in after)
                group_effects.append(dict(group=group,structure_ids=sorted(group_ids[group]),base_mse=x,new_mse=y,relative_error_reduction=(1-y/x if x>1e-12 else None)))
            comparisons.append(dict(comparison=name,kind=kind,base=base,new=new,base_primary=st.mean(primary(e) for e in before),new_primary=st.mean(primary(e) for e in after),mean_paired_relative_error_reduction=st.mean(gains),new_better_seeds=better,new_better_groups=better_groups,base_better_seeds=reverse,base_better_groups=reverse_groups,new_consistent_development_gain=better==3 and better_groups>=4,base_consistent_development_gain=reverse==3 and reverse_groups>=4,paired_initial_weights_verified=True,group_effects=group_effects))
    report=dict(phase=args.phase,primary='equal mean over four horizons of group-macro standardized dynamic MSE',rows=rows,comparisons=comparisons,J_gate_enabled=False,new_independent_test=False,result_sha256={f'{p}:{s}':digest_file(ROOT/'data/manifests'/f'representation_v2_{p}_seed{s}.json') for p in phases for s in (42,43,44)})
    write_json(ROOT/'data/manifests'/f'representation_v2_{args.phase}_summary.json',report)
    lines=[f'# 输入v2：{args.phase}结果','','目标、划分、种子、预算和模型尺寸固定；表中主误差为四个未来间隔等权的组宏平均。仅5个开发验证组，无独立测试。','', '| 模型 | 输入条件 | 主误差均值 ±种子标准差 |','| --- | --- | --- |']
    for r in rows:lines.append(f"| {r['kind']} | {r['profile']} | {r['primary_mse_mean']:.7g} ± {r['primary_mse_seed_std']:.3g} |")
    lines+=['','## 配对贡献','','正收益表示新条件误差降低，负值表示误差增加；drop_static的“新条件”是移除第2/3项，add_geometry的新条件是添加实时几何接触。','', '| 模型 | 比较 | 平均配对误差下降 | 新条件较好种子 | 较好组 | 达到预声明较一致收益规则 |','| --- | --- | --- | --- | --- | --- |']
    for r in comparisons:lines.append(f"| {r['kind']} | {r['comparison']} | {r['mean_paired_relative_error_reduction']:.2%} | {r['new_better_seeds']}/3 | {r['new_better_groups']}/5 | {r['new_consistent_development_gain']} |")
    lines+=['','## 全部未来间隔','','| 模型 | 输入条件 | 未来步数 | 标准化动态MSE | 对线性Skill |','| --- | --- | --- | --- | --- |']
    for r in rows:
        for h in r['horizons']:lines.append(f"| {r['kind']} | {r['profile']} | {h['horizon']} | {h['mse_mean']:.7g} | {h['skill_vs_linear_mean']:.4f} |")
    if args.phase=='geometry':
        lines+=['','## 几何接触增量的逐组差异','','先平均种子的每组主误差，再计算相对下降；正数为改善，负数为退步。与全体平均配对下降的聚合顺序不同，不混算。','', '| 模型 | 条件 | 验证构建体 | 组误差下降 |','| --- | --- | --- | --- |']
        for r in comparisons:
            if r['comparison'].startswith('add_geometry'):
                for g in r['group_effects']:lines.append(f"| {r['kind']} | {r['comparison']} | {','.join(g['structure_ids'])} | {g['relative_error_reduction']:.2%} |")
    lines+=['','初始权重逐种子配对一致，训练查询顺序和参数量一致；移除通道采用标准化后整列置0，不移除索引12/13静态摘要。没有恢复J门控。几何接触只进入输入，预测目标/损失不增加。原量纲逐通道、逐组/种子、线性基线及权重哈希见各JSON。未检测到收益不等于普遍无用，也不等于统计等价。','']
    (ROOT/'docs'/f'REPRESENTATION_V2_{args.phase.upper()}_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
