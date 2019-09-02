import multiprocessing as mp

from rlpyt.envs.dm_control_env import DMControlEnv

def worker(EnvCls, env_kwargs):
    env = EnvCls(**env_kwargs)
    env.reset()

if __name__ == '__main__':
    env_kwargs = dict(domain='cloth_v0', task='easy', max_path_length=1200,
                      task_kwargs=dict(reward='diagonal'),
                      pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=True,
                                                render_kwargs=dict(width=64, height=64)))
    EnvCls = DMControlEnv

    n_processes = 5
    processes = [mp.Process(target=worker, args=(EnvCls, env_kwargs)) for _ in range(n_processes)]
    [p.start() for p in processes]
    [p.join() for p in processes]

    print('Completed')