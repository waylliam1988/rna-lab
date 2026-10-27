"""Summarize all frozen physical-forecast checkpoints without model selection."""
import argparse
import json
import statistics
from acquire_experimental_corpus import ROOT,write_json,digest_file


def aggregate(path):
    report=json.loads(path.read_text());groups={}
    for row in report['results']:
        for history,modes in row['evaluation'].items():
            for mode,metrics in modes.items():
                key=(row['updates'],int(history),mode)
                groups.setdefault(key,[]).append(metrics)
    result=[]
    for (updates,history,mode),rs in sorted(groups.items()):
        values=[r['dynamic_mse'] for r in rs]
        result.append(dict(updates=updates,history=history,mode=mode,seeds=len(rs),dynamic_mse_mean=statistics.mean(values),dynamic_mse_seed_std=statistics.pstdev(values),persistence_dynamic_mse=statistics.mean(r['persistence_dynamic_mse'] for r in rs),mse_by_channel_mean=[statistics.mean(r['group_macro_mse_by_channel'][i] for r in rs) for i in range(5)]))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rollout',type=int,choices=[4,8],required=True);args=parser.parse_args()
    path=ROOT/'data/manifests'/f'physical_forecast_results_r{args.rollout}_v1.json'
    values=aggregate(path)
    summary=dict(rollout=args.rollout,results_sha256=digest_file(path),aggregate=values,new_test_evaluated=False)
    lines=[f'# 物理预测：外部展开{args.rollout}步结果','', '三种子组宏平均动态MSE（Rg/力/能量），±为种子总体标准差。全部数字仅是5条序列/5组的开发验证；所有历史长度预测同一终点。没有选择最佳检查点或评估旧测试。', '', '| 更新次数 | 历史 | 保留状态 MSE | 每步重置 MSE | 持久性 MSE |','| --- | --- | --- | --- | --- |']
    for u in (100,250,500):
        for h in (4,8,16):
            a=next(r for r in values if r['updates']==u and r['history']==h and r['mode']=='carry')
            b=next(r for r in values if r['updates']==u and r['history']==h and r['mode']=='reset')
            lines.append(f"| {u} | {h} | {a['dynamic_mse_mean']:.8g} ± {a['dynamic_mse_seed_std']:.3g} | {b['dynamic_mse_mean']:.8g} | {a['persistence_dynamic_mse']:.8g} |")
    lines+=['','逐通道结果与每种子原始结果见data/manifests对应JSON。输入为已知当前3D状态的原观测；零外部动作、单位逆质量、固定解析力场模拟，不能当作实验轨迹、序列从头折叠或完整原Nezha闭环的成绩。','']
    if args.rollout==8:
        previous=aggregate(ROOT/'data/manifests/physical_forecast_results_r4_v1.json')
        comparisons=[]
        lines+=['## 展开长度对照','','| 共同历史 | 4步×500更新 | 8步×500更新 | 8步×250更新（等输入曝光量） |','| --- | --- | --- | --- |']
        for h in (4,8,16):
            get=lambda rs,u:next(r['dynamic_mse_mean'] for r in rs if r['updates']==u and r['history']==h and r['mode']=='carry')
            a,b,c=get(previous,500),get(values,500),get(values,250)
            comparisons.append(dict(history=h,r4_u500=a,r8_u500=b,r8_u250=c))
            lines.append(f'| {h} | {a:.8g} | {b:.8g} | {c:.8g} |')
        lines+=['','8步×500看到的输入状态数是4步×500的两倍；8步×250与4步×500均16000个输入状态。等曝光量不等于相同计算或梯度路径。只据验证误差不能宣布任何架构的普遍优劣。','']
        summary['rollout_comparisons']=comparisons
    write_json(ROOT/'data/manifests'/f'physical_forecast_summary_r{args.rollout}_v1.json',summary)
    (ROOT/'docs'/f'PHYSICAL_FORECAST_RESULTS_R{args.rollout}_V1.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='aggregate'},indent=2))


if __name__=='__main__':main()
