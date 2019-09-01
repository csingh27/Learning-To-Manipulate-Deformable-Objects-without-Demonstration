
import copy

configs = dict()

config = dict(
    agent=dict(
        v_model_kwargs=None,
        q_model_kwargs=None,
        model_kwargs=None,
    ),
    algo=dict(
        discount=0.99,
        batch_size=256,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=3e-4,
        reparameterize=True,
        policy_output_regularization=0.0,
        reward_scale=1
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=50,
    ),
    env=dict(
        domain='cloth_v0',
        task='easy',
        max_path_length=1200,
        task_kwargs=dict(reward='diagonal')
    ),
    eval_env=dict(
        domain='cloth_v0',
        task='easy',
        max_path_length=1200,
        task_kwargs=dict(reward='diagonal')
    )
)

configs["sac_1M_serial"] = config
