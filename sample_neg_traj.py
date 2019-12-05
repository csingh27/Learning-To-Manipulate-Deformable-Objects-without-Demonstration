from rlpyt.envs.dm_control_env import DMControlEnv
import math
import time
import os
from os.path import join, exists
import itertools
from tqdm import tqdm
import numpy as np
import imageio
import multiprocessing as mp


def worker(worker_id, start, end):
    np.random.seed(worker_id)
    # Initialize environment
    env_args = dict(
        domain='rope_sac',
        task='easy',
        max_path_length=5,
        pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=False, # to not take away non pixel obs
                                  render_kwargs=dict(width=64, height=64, camera_id=0)),
        #task_kwargs=dict(random_location=True, pixels_only=True) # to not return positions and only pick location
    )
    env = DMControlEnv(**env_args)
    if worker_id == 0:
        pbar = tqdm(total=end - start)

    for i in range(start, end):
        str_i = str(i)
        run_folder = join(root, 'run{}'.format(str_i.zfill(5)))
        if not exists(run_folder):
            os.makedirs(run_folder)

        o = env.reset()
        imageio.imwrite(join(run_folder, 'img_{}_{}.png'.format('0'.zfill(2), '0'.zfill(3))), o.pixels.astype('uint8'))
        actions = []
        for t in itertools.count(start=1):
            saved_state = env.get_state()
            str_t = str(t)
            actions_t = []
            for k in range(n_neg_samples):
                str_k = str(k + 1) # start at 1

                a = env.action_space.sample()
                a = a / np.linalg.norm(a) * np.sqrt(2)
                actions_t.append(np.concatenate((o.location[:2], a))) # need to add before b/c o is replaced
                o, _, terminal, info = env.step(a)

                imageio.imwrite(join(run_folder, 'img_{}_{}.png'.format(str_t.zfill(2), str_k.zfill(3))), o.pixels.astype('uint8'))

                env.set_state(saved_state)
                env.step(np.array([0, 0]))
                env._step_count -= 1

            a = env.action_space.sample()
            a = a / np.linalg.norm(a) * np.sqrt(2)
            o, _, terminal, info = env.step(a)

            actions_t.insert(0, np.concatenate((o.location[:2], a)))
            imageio.imwrite(join(run_folder, 'img_{}_{}.png'.format(str_t.zfill(2), '0'.zfill(3))), o.pixels.astype('uint8'))

            actions.append(np.stack(actions_t, axis=0))
            if terminal or info.traj_done:
                break

        actions = np.stack(actions, axis=0)
        np.save(join(run_folder, 'actions.npy'), actions)

        if worker_id == 0:
            pbar.update(1)
    if worker_id == 0:
        pbar.close()

if __name__ == '__main__':
    start = time.time()
    root = join('data', 'rope_data')
    if not exists(root):
        os.makedirs(root)

    n_trajectories = 10
    n_chunks = 1
    n_neg_samples = 100
    #n_chunks = mp.cpu_count()
    partition_size = math.ceil(n_trajectories / n_chunks)
    args_list = []
    for i in range(n_chunks):
        args_list.append((i, i * partition_size, min((i + 1) * partition_size, n_trajectories)))
    print('args', args_list)

    ps = [mp.Process(target=worker, args=args) for args in args_list]
    [p.start() for p in ps]
    [p.join() for p in ps]

    elapsed = time.time() - start
    print('Finished in {:.2f} min'.format(elapsed / 60))
