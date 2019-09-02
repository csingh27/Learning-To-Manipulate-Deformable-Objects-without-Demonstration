
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.conv2d import Conv2dHeadModel, Conv2dModel
from rlpyt.models.preprocessor import get_preprocessor

class PiConvModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            action_size,
            paddings=None,
            nonlinearity=torch.nn.LeakyReLU
            ):
        super().__init__()
        assert all([ks % 2 == 1 for ks in kernel_sizes])
        if paddings is None:
            paddings = [ks // 2 for ks in kernel_sizes]

        self._obs_ndim = 3
        self._action_size = action_size
        self._image_shape = observation_shape.pixels

        self.preprocessor = get_preprocessor('image')
        self.conv = Conv2dHeadModel(observation_shape.pixels, channels, kernel_sizes,
                                    strides, hidden_sizes, output_size=2 * action_size,
                                    paddings=paddings,
                                    nonlinearity=nonlinearity, use_maxpool=False)

    def forward(self, observation, prev_action, prev_reward):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        observation = self.preprocessor(observation)
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        output = self.conv(observation.view(T * B, *self._image_shape))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class QofMuConvModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            action_size,
            paddings=None,
            nonlinearity=torch.nn.LeakyReLU
            ):
        super().__init__()
        assert all([ks % 2 == 1 for ks in kernel_sizes])
        if paddings is None:
            paddings = [ks // 2 for ks in kernel_sizes]

        self._obs_ndim = 3
        self._action_size = action_size
        self._image_shape = observation_shape.pixels

        self.preprocessor = get_preprocessor('image')
        c, h, w = observation_shape.pixels
        self.conv = Conv2dModel(c, channels, kernel_sizes,
                                strides, paddings=paddings, nonlinearity=nonlinearity)
        conv_out_size = self.conv.conv_out_size(h, w)
        self.mlp = MlpModel(conv_out_size + action_size, hidden_sizes,
                            output_size=1, nonlinearity=nonlinearity)

    def forward(self, observation, prev_action, prev_reward, action):
        if isinstance(observation, tuple):
            observation = torch.cat(observation, dim=-1)

        observation = self.preprocessor(observation)
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)

        embedding = self.conv(observation.view(T * B, self._image_shape))
        q_input = torch.cat([embedding, action.view(T * B, -1)], dim=1)
        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q

