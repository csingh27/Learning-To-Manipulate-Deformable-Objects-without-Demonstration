
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/dm_control/qpg/sac/train/dm_control_sac_parallel_v2.py"
affinity_code = encode_affinity(
    n_cpu_core=24,
    n_gpu=8,
    contexts_per_gpu=1,
    n_socket=2,
    hyperthread_offset=24,
)

runs_per_setting = 2
default_config_key = "sac_1M_serial_v2"
experiment_title = "sac_dm_control_parallel_v2_8cpu"
variant_levels = list()

domain = ['cloth_v0']
task = ['easy']
values = list(zip(domain, task))
dir_names = ["env_{}_{}".format(*v) for v in values]
keys = [('env', 'domain'), ('env', 'task')]
variant_levels.append(VariantLevel(keys, values, dir_names))

batch_B = [8, 8, 32, 32]
batch_size = [256, 1024, 256, 1024]
learning_rate = [3e-4, 6e-4, 3e-4, 6e-4]
values = list(zip(batch_B, batch_size, learning_rate))
dir_names = ["batch_B{}_bs{}_lr{}".format(*v) for v in values]
keys = [('sampler', 'batch_B'), ('algo', 'replay_ratio'), ('algo', 'learning_rate')]
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
