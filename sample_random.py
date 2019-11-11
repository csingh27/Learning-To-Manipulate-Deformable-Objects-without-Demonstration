from rlpyt.envs.dm_control_env import DMControlEnv
import os
from os.path import join, exists
import itertools
from tqdm import tqdm
import numpy as np
import imageio

domain = 'rope_sac'
n_samples = 80000
root = join('data', 'rope_data')

env_args = dict(
    domain='rope_sac',
    task='easy',
    max_path_length=100,
    pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=False, # to not take away non pixel obs
                              render_kwargs=dict(width=64, height=64, camera_id=0)),
    #task_kwargs=dict(random_location=True, pixels_only=True) # to not return positions and only pick location
)

if not exists(root):
    os.makedirs(root)

env = DMControlEnv(**env_args)
total = 0

pbar = tqdm(total=n_samples)
cur_episode = 0
while total < n_samples:
    str_i = str(cur_episode)
    run_folder = join(root, 'run{}'.format(str_i.zfill(4)))
    if not exists(run_folder):
        os.makedirs(run_folder)

    actions = []
    o = env.reset()
    for t in itertools.count():
        a = env.action_space.sample()
        actions.append(np.concatenate((o.location[:2], a)))
        str_t = str(t)
        imageio.imwrite(join(run_folder, 'img_{}.png'.format(str_t.zfill(3))), o.pixels.astype('uint8'))
        pbar.update(1)
        total += 1

        o, _, terminal, info = env.step(a)
        if terminal or info.traj_done:
            break

    actions = np.stack(actions, axis=0)
    np.save(join(run_folder, 'actions.npy'), actions)
    cur_episode += 1
