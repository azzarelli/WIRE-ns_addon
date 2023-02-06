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
from typing import Optional, Set, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.field_components.base_field_component import FieldComponent


class RealGaborLayer(nn.Module):
    """Implicit representations for Gabor Nonlinearity"""

    def __init__(self, in_ft, out_ft, bias=False, is_first=False, omega0=10.0, sigma0=10.0, trainable=False) -> None:
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_ft

        self.freqs = nn.Linear(in_ft, out_ft, bias=bias)
        self.scale = nn.Linear(in_ft, out_ft, bias=bias)

    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0

        return torch.cos(omega) * torch.exp(-(scale**2))


class ComplexGaborLayer(nn.Module):
    """
    Implicit representation with complex Gabor nonlinearity

    Inputs;
        in_features: Input features
        out_features; Output features
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega0: Frequency of Gabor sinusoid term
        sigma0: Scaling of Gabor Gaussian term
        trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(self, in_ft, out_ft, bias=True, is_first=False, omega0=10.0, sigma0=40.0, trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_ft

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_ft, out_ft, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())


class WIREMLP(FieldComponent):
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
        activation_type: Optional[str] = "complex_gabor",
        omega: Optional[float] = 10.0,
        sigma: Optional[float] = 10.0,
        trainable=False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width

        if activation_type == "complex_gabor":
            self.nonlin = ComplexGaborLayer
            dtype = torch.cfloat
            self.complex = True
        elif activation_type == "real_gabor":
            self.nonlin = RealGaborLayer
            dtype = torch.float
            self.complex = False
        self.pos_encoder = False

        self.net = []

        self.wavelet = activation_type

        self.build_nn_modules(hidden_layers=num_layers, in_features=in_dim, hidden_features=layer_width,
                                omega=omega, sigma=sigma, 
                                trainable=trainable, dtype=dtype)

    def build_nn_modules(self, dtype=None, in_features:Optional[int]=None, out_features:Optional[int]=None, 
                         hidden_layers:Optional[int]=None, hidden_features:Optional[int]=None,
                         omega:Optional[float]=10.0, sigma:Optional[float]=10.0,
                         trainable:bool=False) -> None:
        """Initialize multi-layer perceptron with no periodic encoding"""
        # First layer
        self.net.append(self.nonlin(in_ft=in_features,
                                    out_ft=hidden_features, 
                                    omega0=omega,
                                    sigma0=sigma,
                                    is_first=True,
                                    trainable=trainable))
        for i in range(hidden_layers):
            self.net.append(self.nonlin(in_ft=hidden_features,
                                        out_ft=hidden_features, 
                                        omega0=omega,
                                        sigma0=sigma))

        print(hidden_features, self.out_dim)
        final_linear = nn.Linear(hidden_features,
                                 self.out_dim,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)

    def forward(self, in_tensor: TensorType["bs":..., "in_dim"]) -> TensorType["bs":..., "out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        output = self.net(in_tensor)

        if self.wavelet == 'gabor':
            return output.real
        return output
