"""Causal forecasting diagnostics; constant channels are excluded explicitly."""
import numpy as np

DYNAMIC_CHANNELS = [0,3,4]


def macro_numpy(observations):
    result=np.array(observations[...,:5],dtype=np.float64,copy=True)
    result[...,0]=np.log1p(result[...,0]);result[...,3]=np.log1p(result[...,3])
    return result


def group_mean(values,metadata):
    groups=sorted({r['split_group_id'] for r in metadata})
    return np.mean([np.mean(values[[i for i,r in enumerate(metadata) if r['split_group_id']==g]],axis=0) for g in groups],axis=0)


def skill(model_mse,baseline_mse,floor=1e-12):
    return None if baseline_mse<=floor else float(1-model_mse/baseline_mse)


def baseline_at_horizon(episodes,horizon,anchor=16):
    if anchor<1 or anchor+horizon>=episodes.shape[1]:raise ValueError('Forecast outside available chronology')
    values=macro_numpy(episodes)
    current=values[:,anchor];target=values[:,anchor+horizon]
    linear=current+horizon*(current-values[:,anchor-1])
    return current,target,linear
