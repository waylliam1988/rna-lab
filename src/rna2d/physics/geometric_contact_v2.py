"""Read current-frame contact diagnostics; no change to force or J dynamics."""
import torch
from .reference import compute_rna_force_field_fast


def geometric_contacts(engine,state,contact,mask):
    _,_,aux=compute_rna_force_field_fast(state.q,contact,mask,engine.ff_params,contact_rc=1.,return_aux=True)
    return torch.stack([aux['contact_mean'],aux['contact_std']],dim=-1)
