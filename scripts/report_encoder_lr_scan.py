"""Train-only acceptance and stop decisions; preserve complete predictions locally."""
import json
import statistics as st
from acquire_experimental_corpus import ROOT,write_json,digest_file


def main():
    p=json.loads((ROOT/'data/manifests/encoder_lr_scan_protocol_v1.json').read_text());reports=[]
    for seed in p['seeds']:
        source=ROOT/'runs'/f'encoder_lr_scan_v1_seed{seed}'/'results.json';full=json.loads(source.read_text())
        assert not full['validation_loaded'] and not full['test_loaded'];assert len(full['results'])==12
        # Full decoded pairs remain immutable in local run artifacts. Public
        # per-RNA metrics suffice without repeating large pair arrays36 times.
        public=json.loads(json.dumps(full))
        for row in public['results']:
            row['details']=[{k:v for k,v in d.items() if k not in ('predicted_pairs','evaluated_predicted_pairs')} for d in row['details']]
        public.update(full_local_result_sha256=digest_file(source),full_pair_predictions_stored_locally=True)
        write_json(ROOT/'data/manifests'/f'encoder_lr_scan_seed{seed}_v1.json',public);reports.append(full)
    aggregate=[];thresholds={};failures=[]
    for kind in p['encoders']:
        kind_finals=[];accepted=[]
        for config in p['variants']:
            rows=[next(r for r in report['results'] if r['encoder']==kind and r['variant']==config['name']) for report in reports]
            good=[r for r in rows if r['status']=='completed'];scores=[r['training']['macro_f1'] for r in good];kind_finals+=scores
            failures += [dict(encoder=kind,variant=config['name'],seed=r['seed'],failure=r['failure']) for r in rows if r['status']!='completed']
            if len(scores)==3 and min(scores)>=.9:accepted.append(config['name'])
            aggregate.append(dict(encoder=kind,variant=config['name'],completed=len(good),train_f1_mean=st.mean(scores) if scores else None,train_f1_seed_std=st.pstdev(scores) if scores else None,min_train_f1=min(scores) if scores else None,max_train_f1=max(scores) if scores else None,seed_scores={str(r['seed']):r['training']['macro_f1'] for r in good},max_logged_train_f1=max((h['train_macro_f1'] for r in rows for h in r['history']),default=None)))
        thresholds[kind]=dict(max_final_train_f1=max(kind_finals) if kind_finals else None,all_final_runs_below0_8=bool(kind_finals) and max(kind_finals)<.8,variants_all_seeds_ge0_9=accepted)
    cnn=[r['cnn_training']['macro_f1'] for r in reports]
    stop=all(t['all_final_runs_below0_8'] for t in thresholds.values()) and not failures
    accepted=min(cnn)>=.9 and all(t['variants_all_seeds_ge0_9'] for t in thresholds.values())
    report=dict(aggregate=aggregate,thresholds=thresholds,cnn_training_seed_scores=dict(zip(map(str,p['seeds']),cnn)),strict_all_below0_8_stop_triggered=stop,validation_acceptance_gate_open=bool(accepted),failures=failures,scan_scope_completed=True,further_tuning_performed=False,validation_loaded=False,test_loaded=False,source_model_unchanged=all(r['model_source_sha256']==p['model_source_sha256'] for r in reports),results_sha256={str(s):digest_file(ROOT/'data/manifests'/f'encoder_lr_scan_seed{s}_v1.json') for s in p['seeds']})
    write_json(ROOT/'data/manifests/encoder_lr_scan_summary_v1.json',report)
    lines=['# 编码器有限lr扫描：仅训练拟合','','先审计EGNN几何输入，再按冻结协议完成3编码器×4变体×3种子，25epoch。所有指标为88条训练RNA的宏F1；本轮没有加载验证/测试文件，不进行新的架构泛化排名。','', '| 编码器 | lr变体 | 训练F1均值 ±种子标准差 | 最小–最大 |','| --- | --- | --- | --- |']
    for r in aggregate:
        if r['train_f1_mean'] is None:lines.append(f"| {r['encoder']} | {r['variant']} | 失败 | 见JSON |");continue
        lines.append(f"| {r['encoder']} | {r['variant']} | {r['train_f1_mean']:.4f} ± {r['train_f1_seed_std']:.4f} | {r['min_train_f1']:.4f}–{r['max_train_f1']:.4f} |")
    lines+=['',f'原CNN参考训练F1：{dict(zip(p["seeds"],cnn))}。没有扫描CNN，也未改变旧检查点。','',f'用户“全部最终运行<0.8”严格止损条件触发：{stop}；所有编码器有共同变体三个种子均≥0.9的验收门槛开放：{bool(accepted)}。失败运行：{len(failures)}。','']
    if stop:lines+=['在三个lr、一个固定warmup和25epoch下，只有原CNN参考达到0.9+；其余默认实现未达到充分拟合门槛。停止调优。本工作不宣称架构优劣，只报告当前配置下CNN是唯一成功充分拟合的编码器。','']
    elif not accepted:lines+=['严格“全部<0.8”条件未触发，但0.9验收门槛仍未满足；用户规定的扫描已结束，停止本轮调优，不继续追加lr/LayerNorm/warmup实验，不进行验证集比较。','']
    else:lines+=['训练拟合验收门槛通过，但本次仍按范围不加载验证集。后续受控验证需明确选取训练内通过的变体；达标不是架构泛化因果归因的充分证明。','']
    lines+=['“未达到拟合门槛”不表示完全没有学习；训练曲线与全部种子公开，不用未达标验证分数给家族排名。逐RNA指标公开，完整解码配对留在本地runs，SHA记录在摘要。Warmup仅前55/550步线性升温，其他设置不变；EGNN位置种子仍非实验RNA几何。','']
    (ROOT/'docs/ENCODER_LR_SCAN_RESULTS_V1.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('aggregate','results_sha256')},indent=2))


if __name__=='__main__':main()
