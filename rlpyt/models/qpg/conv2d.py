
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.conv2d import Conv2dHeadModel, Conv2dModel
from rlpyt.models.preprocessor import get_preprocessor

def _filter_name(fields, name):
    fields = list(fields)
    idx = fields.index(name)
    del fields[idx]
    return fields


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
            nonlinearity=torch.nn.LeakyReLU,
            ):
        super().__init__()
        assert all([ks % 2 == 1 for ks in kernel_sizes])
        if paddings is None:
            # SAME padding for odd kernel sizes
            paddings = [ks // 2 for ks in kernel_sizes]

        self._obs_ndim = 3
        self._action_size = action_size
        self._image_shape = observation_shape.pixels

        self.preprocessor = get_preprocessor('image')

        fields = _filter_name(observation_shape._fields, 'pixels')
        assert all([len(getattr(observation_shape, f)) == 1 for f in fields]), observation_shape
        extra_input_size = sum([getattr(observation_shape, f)[0] for f in fields])
        self.conv = Conv2dHeadModel(observation_shape.pixels, channels, kernel_sizes,
                                    strides, hidden_sizes, output_size=2 * action_size,
                                    paddings=paddings,
                                    nonlinearity=nonlinearity, use_maxpool=False,
                                    extra_input_size=extra_input_size)

    def forward(self, observation, prev_action, prev_reward):
        pixel_obs = self.preprocessor(observation.pixels)
        lead_dim, T, B, _ = infer_leading_dims(pixel_obs, self._obs_ndim)

        pixel_obs = pixel_obs.view(T * B, *self._image_shape)
        fields = _filter_name(observation._fields, 'pixels')
        extra_input = torch.cat([getattr(observation, f).view(T * B, -1)
                                 for f in fields], dim=-1)

        output = self.conv(pixel_obs, extra_input=extra_input)

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
            nonlinearity=torch.nn.LeakyReLU,
            n_tile=1,
            ):
        super().__init__()
        assert all([ks % 2 == 1 for ks in kernel_sizes])
        if paddings is None:
            paddings = [ks // 2 for ks in kernel_sizes]

        self._obs_ndim = 3
        self._action_size = action_size
        self._image_shape = observation_shape.pixels
        self._n_tile = n_tile

        self.preprocessor = get_preprocessor('image')

        fields = _filter_name(observation_shape._fields, 'pixels')
        assert all([len(getattr(observation_shape, f)) == 1 for f in fields]), observation_shape
        extra_input_size = sum([getattr(observation_shape, f)[0] for f in fields])
        self.conv = Conv2dHeadModel(observation_shape.pixels, channels, kernel_sizes,
                                    strides, hidden_sizes, output_size=1,
                                    paddings=paddings,
                                    nonlinearity=nonlinearity, use_maxpool=False,
                                    extra_input_size=extra_input_size + action_size * n_tile)

    def forward(self, observation, prev_action, prev_reward, action):
        pixel_obs = self.preprocessor(observation.pixels)
        lead_dim, T, B, _ = infer_leading_dims(pixel_obs, self._obs_ndim)

        pixel_obs = pixel_obs.view(T * B, *self._image_shape)
        fields = _filter_name(observation._fields, 'pixels')
        action = action.view(T * B, -1).repeat(1, self._n_tile)
        extra_input = torch.cat([getattr(observation, f).view(T * B, -1)
                                 for f in fields] + [action], dim=-1)

        q = self.conv(pixel_obs, extra_input=extra_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)

        return q

