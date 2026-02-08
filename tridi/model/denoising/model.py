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
        dim_second_sbj: int,
        dim_timestep_embed: int,
        dim_output: int,
        cond_channels: int = 0,
        **kwargs
    ):
        super().__init__()

        self.name = name
        if self.name == 'transformer_unidiffuser_3':
            self.autocast_context = torch.autocast('cuda', dtype=torch.float32)
            self.model = TransformertUni3WayModel(
                dim_sbj=dim_sbj, 
                dim_second_sbj=dim_second_sbj,
                dim_timestep_embed=dim_timestep_embed,
                dim_output=dim_output,
                cond_channels=cond_channels,
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
            data_dim = self.model.dim_sbj + self.model.dim_second_sbj
            assert inputs.shape[1] >= data_dim, f"inputs dim {inputs.shape[1]} < data_dim {data_dim}"

            # print("Using unidiffuser 3 way model")
            with self.autocast_context:
                data = inputs[:, :data_dim]
                cond = inputs[:, data_dim:] if inputs.shape[1] > data_dim else None
                if cond is not None and self.model.cond_channels > 0:
                    assert cond.shape[1] == self.model.cond_channels, \
                        f"cond dim mismatch: got {cond.shape[1]} vs expected {self.model.cond_channels}"

                sbj, second_sbj = torch.split(
                    data,
                    [self.model.dim_sbj, self.model.dim_second_sbj],  # ✅ 修正
                    dim=1
                )
                return self.model(sbj, second_sbj, t, t_second, cond)
        else:
            with self.autocast_context:
                return self.model(inputs, t)
