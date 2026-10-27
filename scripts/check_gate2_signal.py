"""Gate 2: measure change before fitting; fixed anchor and all four horizons."""
import json
import sys
import numpy as np
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
from rna2d.physics.diagnostics import baseline_at_horizon,macro_numpy,group_mean,skill,DYNAMIC_CHANNELS as C


def main():
    output=ROOT/'data/manifests/gate2_signal_v1.json'
    if output.exists():raise ValueError('Gate already frozen')
    data=ROOT/'data/experimental/physical_forecast_v1'
    train=np.load(data/'train.npz')['observations'];scale=macro_numpy(train).std(axis=(0,1)).clip(1e-4)
    report=dict(anchor=16,horizons=[1,4,8,16],normalization='training pool only; macro channel std floor1e-4',scale=scale.tolist(),source_sha256={s:digest_file(data/(s+'.npz')) for s in ('train','validation')},script_sha256=digest_file(Path(__file__)),splits={},new_test_evaluated=False)
    lines=['# 第二关：状态变化与预测间隔','','固定当前时刻16，预测未来1/4/8/16步。无模型拟合或超参数选择；模拟观测不是实验轨迹。','', '| 集合 | 未来步数 | 持久性动态MSE | 线性外推动力MSE | 线性Skill | 超数值容差的动态变化比例 |','| --- | --- | --- | --- | --- | --- |']
    for split in ('train','validation'):
        episodes=np.load(data/(split+'.npz'))['observations'];meta=json.loads((data/(split+'_metadata.json')).read_text());results=[]
        for h in report['horizons']:
            current,target,linear=baseline_at_horizon(episodes,h)
            delta=target-current;persist=(delta**2);error=(linear-target)**2
            p=group_mean(persist,meta);e=group_mean(error,meta)
            # This is a conservative float32 precision heuristic, NOT experimental noise.
            floor=32*np.finfo(np.float32).eps*np.maximum(1,np.maximum(np.abs(current),np.abs(target)))
            fraction=group_mean((np.abs(delta)>floor).astype(float),meta)
            pnorm=float(np.mean(group_mean(persist/scale**2,meta)[C]));enorm=float(np.mean(group_mean(error/scale**2,meta)[C]))
            r=dict(horizon=h,delta_abs_mean_by_channel=group_mean(np.abs(delta),meta).tolist(),delta_rms_by_channel=np.sqrt(p).tolist(),persistence_mse_by_channel=p.tolist(),linear_mse_by_channel=e.tolist(),persistence_dynamic_mse=float(p[C].mean()),linear_dynamic_mse=float(e[C].mean()),standardized_persistence_mse=pnorm,standardized_linear_mse=enorm,linear_standardized_skill=skill(enorm,pnorm),linear_raw_skill=skill(float(e[C].mean()),float(p[C].mean())),fraction_changes_above_float32_heuristic=fraction.tolist())
            results.append(r)
            lines.append(f"| {split} | {h} | {r['persistence_dynamic_mse']:.8g} | {r['linear_dynamic_mse']:.8g} | {r['linear_standardized_skill']:.5f} | {np.mean(fraction[C]):.3%} |")
        report['splits'][split]=results
    report['constant_contact_channels_verified']=all(max(r['delta_abs_mean_by_channel'][i] for i in (1,2))==0 for rs in report['splits'].values() for r in rs)
    assert report['constant_contact_channels_verified']
    lines+=['','Skill使用训练通道标准差归一化后、按组平均的动态MSE计算；表中原量纲动态MSE仅用于与旧结果对应，二者不能混算。数值容差仅为32×float32 epsilon×量级，不是实测噪声或生物学效应阈值。','', '两项接触通道变化恒为0，排除出动态主指标。详细逐通道变化量、误差及原量纲Skill见JSON。残差写法本身不会改变预测难度：零残差等于持久性；必须进一步胜过线性变化趋势。','']
    write_json(output,report)
    (ROOT/'docs/GATE2_SIGNAL_V1.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(dict(constant_contact_channels_verified=report['constant_contact_channels_verified'],validation=report['splits']['validation']),indent=2))


if __name__=='__main__':main()
