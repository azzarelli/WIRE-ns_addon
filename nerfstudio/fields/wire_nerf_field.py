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

"""Wire NeRF field"""


from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.field_components.wire_mlp import WIREMLP as WMLP
from nerfstudio.fields.base_field import Field


class WIREField(Field):
    """Wire NeRF Field

    Args:
        tx_mlp_num_layers: Number of layers for tx MLP.
        tx_mlp_layer_width: Width of tx MLP layers.
        colour_mlp_num_layers: Number of layer for output colour MLP.
        colour_mlp_layer_width: Width of output colour MLP layers.
        skip_connections: Where to add skip connection in tx MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
        self,
        input_dims: int = 3,

        trainable:bool=False,

        tx_mlp_num_layers: int = 4,
        tx_mlp_layer_width: int = 184,

        tx_sigma: float = 10.0,
        tx_omega: float = 10.0,
        
        colour_mlp_num_layers: int = 4,
        colour_mlp_layer_width: int = 184,
        colour_sigma: float = 10.0,
        colour_omega: float = 10.0,

        tx_mlp_activation_type: str = 'complex_gabor',
        colour_mlp_activation_type: str = 'complex_gabor',
        spatial_distortion: Optional[SpatialDistortion] = None,

        field_colours: Tuple[FieldHead] = (RGBFieldHead(),),

    ) -> None:
        super().__init__()

        self.spatial_distortion = spatial_distortion

        self.mlp_tx = WMLP(
            in_dim=input_dims,
            num_layers=tx_mlp_num_layers,
            layer_width=tx_mlp_layer_width,
            activation_type=tx_mlp_activation_type,
            omega= tx_omega,
            sigma = tx_sigma,
            trainable=trainable
        )

        self.mlp_colour = WMLP(
            in_dim=input_dims+3,
            num_layers=colour_mlp_num_layers,
            layer_width=colour_mlp_layer_width,
            activation_type=colour_mlp_activation_type,
            omega= colour_omega,
            sigma = colour_sigma,
            trainable=trainable
        )


        self.field_output_density = DensityFieldHead(in_dim=self.mlp_tx.get_out_dim())
        self.field_colours = nn.ModuleList(field_colours)
        for field_colour in self.field_colours:
            field_colour.set_in_dim(self.mlp_colour.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples):
        tx_mlp_out = self.mlp_tx(ray_samples.frustums.get_positions())
        density = self.field_output_density(tx_mlp_out.real)
        return density, tx_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        for field_colour in self.field_colours:
            # encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            # print(ray_samples.frustums.get_positions().shape, ray_samples.frustums.)
            mlp_out = self.mlp_colour(torch.cat([ray_samples.frustums.get_positions(), ray_samples.frustums.directions], dim=-1) ) #torch.cat([density_embedding.real], dim=-1))  # type: ignore
            outputs[field_colour.field_head_name] = field_colour(mlp_out.real)
        return outputs
