
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

config = dict(
    agent=dict(
        q_model_kwargs=dict(hidden_sizes=[256, 256]),
        model_kwargs=dict(hidden_sizes=[256, 256]),
    ),
    algo=dict(
        discount=0.99,
        batch_size=256,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=3e-4,
        reparameterize=True,
        policy_output_regularization=0.0,
        reward_scale=1,
        replay_ratio=256,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=5e5,
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

configs["sac_1M_serial_v2"] = config

config = dict(
    agent=dict(
        ModelCls='PiConvModel',
        QModelCls='QofMuConvModel',
        q_model_kwargs=dict(image_shape=(3, 64, 64), channels=(64, 64, 64),
                            kernel_sizes=(3, 3, 3), strides=(2, 2, 2),
                            hidden_sizes=[256, 256]),
        model_kwargs=dict(image_shape=(3, 64, 64), channels=(64, 64, 64),
                          kernel_sizes=(3, 3, 3), strides=(2, 2, 2),
                          hidden_sizes=[256, 256]),
    ),
    algo=dict(
        discount=0.99,
        batch_size=1024,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=6e-4,
        reparameterize=True,
        policy_output_regularization=0.0,
        reward_scale=1,
        replay_ratio=128,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=5e5,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=32,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=50,
    ),
    env=dict(
        domain='cloth_v0',
        task='easy',
        max_path_length=1200,
        task_kwargs=dict(reward='diagonal'),
        pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=True,
                                  render_kwargs=dict(width=64, height=64))
    ),
    eval_env=dict(
        domain='cloth_v0',
        task='easy',
        max_path_length=1200,
        task_kwargs=dict(reward='diagonal'),
        pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=True,
                                  render_kwargs=dict(width=64, height=64))
    )
)

configs["sac_parallel_pixels_v2"] = config

