
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.distributions.categorical import Categorical, DistInfo


MIN_LOG_STD = -20
MAX_LOG_STD = 2


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


GumbelDistInfo = namedtuple('GumbelDistInfo', ['cat_dist', 'delta_dist'])
class GumbelPiMlpModel(torch.nn.Module):
    """For picking corners"""

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            all_corners=False
            ):
        super().__init__()
        self._obs_ndim = 1
        self._all_corners = all_corners
        input_dim = int(np.sum(observation_shape))

        print('all corners', self._all_corners)
        delta_dim = 12 if all_corners else 3
        self.mlp = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=2 * delta_dim + 4, # 3 for each corners, times two for std, 4 probs
        )

        self.delta_distribution = Gaussian(
            dim=delta_dim,
            squash=True,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )
        self.cat_distribution = Categorical(4)


    def forward(self, observation, prev_action, prev_reward):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        output = self.mlp(observation.view(T * B, -1))
        prob = F.softmax(output[:, :4] / 10., dim=-1)
        mu, log_std = output[:, 4:4 + 3], output[:, 4 + 3:]
        prob, mu, log_std = restore_leading_dims((prob, mu, log_std), lead_dim, T, B)
        return GumbelDistInfo(cat_dist=DistInfo(prob=prob), delta_dist=DistInfoStd(mean=mu, log_std=log_std))

    def sample_loglikelihood(self, dist_info):
        cat_dist_info, delta_dist_info = dist_info.cat_dist, dist_info.delta_dist

        cat_sample, cat_loglikelihood = self.cat_distribution.sample_loglikelihood(cat_dist_info)
        one_hot = torch.zeros_like(cat_dist_info.prob)
        cat_sample = cat_sample.unsqueeze(-1)
        one_hot.scatter_(1, cat_sample, 1)
        one_hot = (one_hot - cat_dist_info.prob).detach() + cat_dist_info.prob # Make action differentiable through prob

        if self._all_corners:
            mu, log_std = delta_dist_info.mean, delta_dist_info.log_std
            mu, log_std = mu.view(mu.shape[0], 4, 3), log_std.view(log_std.shape[0], 4, 3)
            mu = mu[torch.arange(len(cat_sample)), cat_sample.squeeze(-1)]
            log_std = log_std[torch.arange(len(cat_sample)), cat_sample.squeeze(-1)]
            new_dist_info = DistInfoStd(mean=mu, log_std=log_std)
        else:
            new_dist_info = delta_dist_info

        delta_sample, delta_loglikelihood = self.delta_distribution.sample_loglikelihood(new_dist_info)
        action = torch.cat((one_hot, delta_sample), dim=-1)
        log_likelihood = cat_loglikelihood + delta_loglikelihood
        return action, log_likelihood

    def sample(self, dist_info):
        cat_dist_info, delta_dist_info = dist_info.cat_dist, dist_info.delta_dist
        if self.training:
            cat_sample = self.cat_distribution.sample(cat_dist_info)
        else:
            cat_sample = torch.max(cat_dist_info.prob, dim=-1)[1].view(-1)
        cat_sample = cat_sample.unsqueeze(-1)
        one_hot = torch.zeros_like(cat_dist_info.prob)
        one_hot.scatter_(-1, cat_sample, 1)

        if self._all_corners:
            mu, log_std = delta_dist_info.mean, delta_dist_info.log_std
            mu, log_std = mu.view(mu.shape[0], 4, 3), log_std.view(log_std.shape[0], 4, 3)
            mu = mu[torch.arange(len(cat_sample)), cat_sample.squeeze(-1)]
            log_std = log_std[torch.arange(len(cat_sample)), cat_sample.squeeze(-1)]
            new_dist_info = DistInfoStd(mean=mu, log_std=log_std)
        else:
            new_dist_info = delta_dist_info

        if self.training:
            self.delta_distribution.set_std(None)
        else:
            self.delta_distribution.set_std(0)
        delta_sample = self.delta_distribution.sample(new_dist_info)
        return torch.cat((one_hot, delta_sample), dim=-1)


class GumbelAutoregPiMlpModel(torch.nn.Module):
    """For picking corners autoregressively"""

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            n_tile=20,
    ):
        super().__init__()
        self._obs_ndim = 1
        self._n_tile = n_tile
        input_dim = int(np.sum(observation_shape))

        self._action_size = action_size
        self.mlp_loc = MlpModel(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=4
        )
        self.mlp_delta = MlpModel(
            input_size=input_dim + 4 * n_tile,
            hidden_sizes=hidden_sizes,
            output_size=3 * 2,
        )

        self.delta_distribution = Gaussian(
            dim=3,
            squash=True,
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )
        self.cat_distribution = Categorical(4)

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
            prob = F.softmax(self.mlp_loc(input_obs), -1)
            prob = restore_leading_dims(prob, lead_dim, T, B)
            self._counter += 1
            return DistInfo(prob=prob)

        elif self._counter == 1:
            assert len(actions) == 1
            action_loc = actions[0].view(T * B, -1)
            model_input = torch.cat((input_obs, action_loc.repeat((1, self._n_tile))), dim=-1)
            output = self.mlp_delta(model_input)
            mu, log_std = output.chunk(2, dim=-1)
            mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
            self._counter += 1
            return DistInfoStd(mean=mu, log_std=log_std)
        else:
            raise Exception('Invalid self._counter', self._counter)

    def has_next(self):
        return self._counter < 2

    def sample_loglikelihood(self, dist_info):
        cat_dist_info, delta_dist_info = dist_info.cat_dist, dist_info.delta_dist

        cat_sample, cat_loglikelihood = self.cat_distribution.sample_loglikelihood(cat_dist_info)
        one_hot = torch.zeros_like(cat_dist_info.prob)
        cat_sample = cat_sample.unsqueeze(-1)
        one_hot.scatter_(1, cat_sample, 1)
        one_hot = (one_hot - cat_dist_info.prob).detach() + cat_dist_info.prob  # Make action differentiable through prob

        delta_sample, delta_loglikelihood = self.delta_distribution.sample_loglikelihood(delta_dist_info)
        action = torch.cat((one_hot, delta_sample), dim=-1)
        log_likelihood = cat_loglikelihood + delta_loglikelihood
        return action, log_likelihood

    def sample(self, dist_info):
        cat_dist_info, delta_dist_info = dist_info.cat_dist, dist_info.delta_dist
        if self.training:
            cat_sample = self.cat_distribution.sample(cat_dist_info)
        else:
            cat_sample = torch.max(cat_dist_info.prob, dim=-1)[1].view(-1)
        cat_sample = cat_sample.unsqueeze(-1)
        one_hot = torch.zeros_like(cat_dist_info.prob)
        one_hot.scatter_(-1, cat_sample, 1)

        if self.training:
            self.delta_distribution.set_std(None)
        else:
            self.delta_distribution.set_std(0)
        delta_sample = self.delta_distribution.sample(delta_dist_info)
        return torch.cat((one_hot, delta_sample), dim=-1)



class QofMuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,
            n_tile=1,
            ):
        super().__init__()
        self._obs_ndim = 1
        self._n_tile = n_tile
        input_dim = int(np.sum(observation_shape))

        # self._obs_ndim = len(observation_shape)
        # input_dim = int(np.prod(observation_shape))

        input_dim += action_size * n_tile
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
        action = action.view(T * B, -1).repeat(1, self._n_tile)
        q_input = torch.cat(
            [observation.view(T * B, -1), action], dim=1)
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
