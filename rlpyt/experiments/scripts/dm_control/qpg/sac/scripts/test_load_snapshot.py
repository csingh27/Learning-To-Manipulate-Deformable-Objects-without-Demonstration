
from os.path import join
import argparse
import json

import torch

from rlpyt.envs.dm_control_env import DMControlEnv
from rlpyt.agents.qpg.sac_agent_v2 import SacAgent
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.serial.sampler import SerialSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot_dir', type=str)
    args = parser.parse_args()

    snapshot_file = join(args.snapshot_dir, 'params.pkl')
    config_file = join(args.snapshot_dir, 'params.json')

    params = torch.load(snapshot_file)
    with open(config_file, 'r') as f:
        config = json.load(f)

    itr, cum_steps = params['itr'], params['cum_steps']
    print(f'Loading experiment at itr {itr}, cum_steps {cum_steps}')
    agent_state_dict = params['agent_state_dict']
    optimizer_state_dict = params['optimizer_state_dict']

    agent = SacAgent(**config["agent"])
    agent.load_state_dict(agent_state_dict)

    sampler = SerialSampler(
        EnvCls=DMControlEnv,
        env_kwargs=config["env"],
        CollectorCls=CpuResetCollector,
        eval_env_kwargs=config["eval_env"],
        **config["sampler"]
    )
    sampler.initialize(agent)

    traj_infos = sampler.evaluate_agent(0, include_observations=True)
    import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    main()