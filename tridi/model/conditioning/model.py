from typing import Optional, Union

from diffusers import ModelMixin
import torch
import torch.nn.functional as F
from torch import Tensor


class ConditioningModel(ModelMixin):
    def __init__(
        self,
        # class
        use_class_conditioning: bool = False,
        num_classes: int = 2,
        # pointnext encoding for object
        use_pointnext_conditioning: bool = False,
        # contacts - for 3way unidiffuser
        use_contacts: str = "",
        contact_model: str = "",  # for compatibility
    ):
        super().__init__()
        # Types of conditioning
        self.use_class_conditioning = use_class_conditioning
        self.use_pointnext_conditioning = use_pointnext_conditioning
        self.use_contacts = use_contacts
        # Number of object classes
        self.num_classes = num_classes

        # Additional input dimensions for conditioning
        self.cond_channels = 0
        if self.use_class_conditioning:
            self.cond_channels += self.num_classes
        if self.use_pointnext_conditioning:
            self.cond_channels += 1024  # length of a feature vector

    def get_input_with_conditioning(
        self,
        x_t: Tensor,
        t: Optional[Tensor] = None,
        t_aux: Optional[Tensor] = None,  # second timestep for unidiffuser
    ):
        # Get dimensions
        B, N = x_t.shape[:2]
        
        # Initial input is the point locations
        x_t_input = [x_t]
        x_t_cond = []

        # # dropping conditioning for regularization
        # # check train / eval flag
        # x_t_cond = torch.cat(x_t_cond, dim=1)  # (B, D_cond)
        # if self.training and torch.rand(1) < 0.1:
        #     x_t_cond = torch.zeros_like(x_t_cond)

        # Concatenate together all the features
        _input = torch.cat([*x_t_input, *x_t_cond], dim=1)  # (B, D)

        return _input
