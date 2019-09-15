
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/dm_control/qpg/sac/train/dm_control_sac.py"
affinity_code = encode_affinity(
    n_cpu_core=24,
    n_gpu=6,
    contexts_per_gpu=1,
)

runs_per_setting = 2
default_config_key = "sac_state_rope_v2"
experiment_title = "sac_state_rope_v2_all"
variant_levels = list()

# exp 1: simultaneous pick / place
# exp 2: conditional pick / place
# exp 3: random loc, learned place

domain = ['rope_v2'] * 3
task = ['easy'] * 3
model_cls = ['PiMlpModel', 'AutoregPiMlpModel', 'PiMlpModel']
random_location = [False, False, True]
sac_module = ['sac_v2'] * 3
sac_agent_module = ['sac_agent_v2', 'sac_agent_autoreg_v2', 'sac_agent_v2']
values = list(zip(domain, task, model_cls, random_location, sac_module, sac_agent_module))
dir_names = ["env_{}_{}_modelcls_{}_rnd_loc_{}".format(*v) for v in values]
keys = [('env', 'domain'), ('env', 'task'), ('agent', 'ModelCls'), ('env', 'task_kwargs', 'random_location'),
        ('sac_module',), ('sac_agent_module',)]
variant_levels.append(VariantLevel(keys, values, dir_names))
variants, log_dirs = make_variants(*variant_levels)

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
