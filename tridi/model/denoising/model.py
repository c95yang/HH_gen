from typing import Optional

import torch
from diffusers import ModelMixin
from torch import Tensor

from tridi.model.denoising.transformer_uni_3 import TransformertUni3WayModel


class DenoisingModel(ModelMixin):
    """Model that denoises parameters at each diffusion step"""
    def __init__(
        self,
        name: str,
        dim_sbj: int,
        dim_timestep_embed: int,
        dim_output: int,
        **kwargs
    ):
        super().__init__()

        self.name = name
        if self.name == 'transformer_unidiffuser_3':
            self.autocast_context = torch.autocast('cuda', dtype=torch.float32)
            self.model = TransformertUni3WayModel(
                dim_sbj=dim_sbj, 
                dim_timestep_embed=dim_timestep_embed,
                dim_output=dim_output,
                **kwargs
            )
        else:
            raise NotImplementedError('Unknown DenoisingModel type: {}'.format(self.name))

    def forward(
            self,
            inputs: Tensor,
            t: Tensor,
            t_second: Optional[Tensor] = None
    ) -> Tensor:
        """ Receives input of shape (B, in_channels) and returns output
            of shape (B, out_channels) """
        if self.name.endswith('unidiffuser_3'):
            with self.autocast_context:
                sbj, second_sbj = torch.split(
                    inputs,
                    [self.model.dim_sbj, self.model.dim_sbj],
                    dim=1
                )
            return self.model(sbj, second_sbj, t, t_second)
        else:
            with self.autocast_context:
                return self.model(inputs, t)
