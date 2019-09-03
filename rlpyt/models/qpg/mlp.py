
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel


class MuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            output_max=1,
            ):
        super().__init__()
        self._output_max = output_max
        self._obs_ndim = len(observation_shape)
        input_dim = int(np.prod(observation_shape))
        self.mlp = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_size,
        )

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        mu = self._output_max * torch.tanh(self.mlp(observation.view(T * B, -1)))
        mu = restore_leading_dims(mu, lead_dim, T, B)
        return mu


class PiMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            ):
        super().__init__()
        self._obs_ndim = 1
        input_dim = int(np.sum(observation_shape))

        # self._obs_ndim = len(observation_shape)
        # input_dim = int(np.prod(observation_shape))

        self._action_size = action_size
        self.mlp = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )

    def forward(self, observation, prev_action, prev_reward):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        output = self.mlp(observation.view(T * B, -1))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class AutoregPiMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            n_tile=50,
    ):
        super().__init__()
        self._obs_ndim = 1
        input_dim = int(np.sum(observation_shape))
        self._n_tile = n_tile

        # self._obs_ndim = len(observation_shape)
        # input_dim = int(np.prod(observation_shape))

        assert action_size == 5 # First 2 (location), then 3 (action)
        self._action_size = action_size

        self.mlp_loc = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=2 * 2
        )
        self.mlp_force = MlpModel(
            input_size=input_dim + 2 * n_tile,
            hidden_sizes=hidden_sizes,
            output_size=3 * 2,
        )

        self._counter = 0

    def start(self):
        self._counter = 0

    def next(self, actions, observation, prev_action, prev_reward):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
                                               self._obs_ndim)
        input_obs = observation.view(T * B, -1)
        if self._counter == 0:
            output = self.mlp_loc(input_obs)
            mu, log_std = output[:, :2], output[:, 2:]
        elif self._counter == 1:
            assert len(actions) == 1
            action_loc = actions[0].view(T * B, -1)
            model_input = torch.cat((input_obs, action_loc.repeat((1, self._n_tile))), dim=-1)
            output = self.mlp_force(model_input)
            mu, log_std = output[:, :3], output[:, 3:]
        else:
            raise Exception('Invalid self._counter', self._counter)
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        self._counter += 1
        return mu, log_std

    def has_next(self):
        return self._counter < 2


class QofMuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            ):
        super().__init__()
        self._obs_ndim = 1
        input_dim = int(np.sum(observation_shape))

        # self._obs_ndim = len(observation_shape)
        # input_dim = int(np.prod(observation_shape))

        input_dim += action_size
        self.mlp = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward, action):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        q_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q


class VMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size=None,  # Unused but accept kwarg.
            ):
        super().__init__()
        self._obs_ndim = 1
        input_dim = int(np.sum(observation_shape))

        # self._obs_ndim = len(observation_shape)
        # input_dim = int(np.prod(observation_shape))

        self.mlp = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, prev_action, prev_reward):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        v = self.mlp(observation.view(T * B, -1)).squeeze(-1)
        v = restore_leading_dims(v, lead_dim, T, B)
        return v
