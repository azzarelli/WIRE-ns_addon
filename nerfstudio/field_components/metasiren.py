# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi Layer Perceptron
"""
from collections import OrderedDict
from typing import Optional, Set, Tuple

import torch
from torch import nn
from torchmeta.modules import MetaModule, MetaSequential
from torchtyping import TensorType

from nerfstudio.field_components.base_field_component import FieldComponent


class MetaCustomLinear(nn.Linear, MetaModule):
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.linear(input, params['weight'], bias) if 'weight_orig' not in params.keys() \
            else F.linear(input, params['weight_orig']*self.weight_mask, bias)

class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0*x)

class MetaSiren(MetaModule):
    """
    Single layer of SIREN; uses SIREN-style init. scheme.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False, is_final=False):
        super().__init__()
        # Encapsulates MetaLinear and activation.
        self.linear = MetaLinear(dim_in, dim_out)
        self.activation = nn.Identity() if is_final else Sine(w0)
        # Initializes according to SIREN init.
        self.init_(c=c, w0=w0, is_first=is_first)

    def init_(self, c, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1/dim_in if is_first else (math.sqrt(c/dim_in)/w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x, params=None):
        return self.activation(self.linear(x, self.get_subdict(params, 'linear')))

class MetaSirenNet(FieldComponent):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        w0:Optional[float]=30., w0_initial:Optional[float]=30.
    ) -> None:

        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []

        for i in range(self.num_layers ):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(MetaSiren(dim_in=layer_dim_in, dim_out=dim_hidden, w0=layer_w0, is_first=is_first))
        layers.append(MetaSiren(dim_in=dim_hidden, dim_out=dim_out, w0=w0, is_final=True))

        self.layers = MetaSequential(*layers)
        # self.layers = nn.ModuleList(layers)

    def forward(self, in_tensor: TensorType["bs":..., "in_dim"], params=None) -> TensorType["bs":..., "out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        return self.layers(x, params=self.get_subdict(params, 'layers'))
