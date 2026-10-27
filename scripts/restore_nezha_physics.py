"""Extract audited definitions only; never import or mutate the old application."""
import ast
import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('D:/Nezha/Nezha-bio')


def extract(source, names):
    tree = ast.parse(source)
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    lines = source.splitlines()
    result = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        result.append('\n'.join(lines[start-1:node.end_lineno]))
    return result


def main():
    output = ROOT / 'src/rna2d/physics'
    output.mkdir(exist_ok=True)
    sources = {name: (OLD / name).read_text(encoding='utf-8-sig') for name in ('config.py', 'kernels.py', 'ghost.py', 'rna_system.py')}
    tree = ast.parse(sources['rna_system.py'])
    bio = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BioSystem')
    methods = ['_fold_precompute_sums_and_contacts', '_fold_build_obs_vec', '_get_grad_gate_scale', '_RMSGradPrecondition', '_apply_grad_gate', '_fold_update_hallucination_loss', '_get_nff_j_mix', '_langevin_step_simple']
    method_source = '\n\n'.join(extract(textwrap.dedent('\n'.join(sources['rna_system.py'].splitlines()[bio.lineno:bio.end_lineno])), methods))
    selections = {
        'kernels.py': ['RWKVCellLite', 'ProtocolSpec', '_CouplingMLP', 'USBLink', 'smooth_force_bound_jit', 'scale_forces_scalar_clip_jit', 'jit_nose_hoover_step', 'jit_integrate_trajectory'],
        'ghost.py': ['ContinuousControlLayer', 'NeuralCortex', 'Ghost'],
        'rna_system.py': ['RNAForceFieldParams', '_make_seqsep_mask', '_energy_norm_denom', '_energy_norm_mode_from_config', '_normalize_energy_per_sample', '_masked_mean', '_masked_center', 'compute_rna_force_field_fast', 'residue_invariants', 'project_orientation'],
    }
    bodies = {name: '\n\n'.join(extract(sources[name], names)) for name, names in selections.items()}
    all_code = '\n'.join(bodies.values()) + '\n' + method_source
    parsed = ast.parse(all_code)
    config_names = {n.attr for n in ast.walk(parsed) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'Config'}
    config_names.update(n.args[1].value for n in ast.walk(parsed) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'getattr' and len(n.args)>1 and isinstance(n.args[0], ast.Name) and n.args[0].id == 'Config' and isinstance(n.args[1], ast.Constant))
    # These names are passed indirectly to the exact gradient-gate method.
    config_names.update(['RNA_OBS_ENERGY_GRAD_MAX_SCALE', 'RNA_OBS_ENERGY_GRAD_WARMUP_STEPS', 'RNA_OBS_ENERGY_GRAD_INFER_FULL_SCALE'])
    config_names.update(['RNA_BP_R0','RNA_BP_SIGMA','RNA_BP_GAIN','RNA_BP_TAIL_ENABLE','RNA_BP_TAIL_SIGMA','RNA_BP_TAIL_GAIN','RNA_REP_R','RNA_REP_SOFT','RNA_REP_GAIN','RNA_REP_POWER','RNA_BOND_R0','RNA_BOND_K','RNA_STACK_K','RNA_BEND_K','RNA_MIN_SEQ_SEP','RNA_ENERGY_NORM','RNA_DT','RNA_GAMMA_DYN','RNA_TARGET_TEMP','RNA_CONTACT_RC','RNA_PORT_GAIN','RNA_GHOST_FORCE_CLIP','RNA_XI_DETACH'])
    config_class = next(n for n in ast.parse(sources['config.py']).body if isinstance(n, ast.ClassDef) and n.name == 'Config')
    assignments = {}
    for node in config_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in config_names:
                    try: assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        if target.id != 'GHOST_DEVICE': raise ValueError('Nonliteral configuration: ' + target.id)
    # Device is an execution choice; the numeric/model settings remain original.
    assignments['GHOST_DEVICE'] = 'cpu'
    config_text = '"""Frozen literal subset of original Config; no legacy imports."""\nclass Config:\n' + ''.join(f'    {k} = {v!r}\n' for k,v in sorted(assignments.items()))
    header = '"""Verbatim AST-selected Nezha definitions; see restoration manifest."""\nimport math\nfrom dataclasses import dataclass\nfrom typing import Dict, Optional, Tuple\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\nfrom .config import Config\n\n'
    contents = {
        '__init__.py': '"""Audited physical interfaces restored from Nezha, with explicit scope."""\n',
        'config.py': config_text,
        'kernels.py': header + bodies['kernels.py'] + '\n',
        'ghost.py': header + 'from .kernels import RWKVCellLite, ProtocolSpec, USBLink\n\n' + bodies['ghost.py'] + '\n',
        'reference.py': header + 'from .kernels import jit_integrate_trajectory\n\n' + bodies['rna_system.py'] + '\n\nclass BioSystem:\n' + textwrap.indent(method_source, '    ') + '\n',
    }
    for name, content in contents.items(): (output/name).write_text(content, encoding='utf-8', newline='\n')
    manifest = {'source_sha256': {k: hashlib.sha256((OLD/k).read_bytes()).hexdigest() for k in sources}, 'selected_top_level': selections, 'selected_biosystem_methods': methods, 'literal_config': assignments, 'missing_config_uses_original_getattr_defaults': sorted(config_names - assignments.keys()), 'generated_sha256': {k:hashlib.sha256(v.encode()).hexdigest() for k,v in contents.items()}, 'scope': 'Exact selected definitions; full BioSystem trainer, perception/policy heads and learned checkpoints are not restored.'}
    (ROOT/'data/manifests/nezha_physics_restoration_v1.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({'generated_files': list(contents), 'config_count':len(assignments), 'fallback_names':manifest['missing_config_uses_original_getattr_defaults']}, indent=2))


if __name__ == '__main__':
    main()
