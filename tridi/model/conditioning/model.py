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
        # NEW
        use_gender_conditioning: bool = False,
        num_genders: int = 2,  # male/female
    ):
        super().__init__()
        # Types of conditioning
        self.use_class_conditioning = use_class_conditioning
        self.use_pointnext_conditioning = use_pointnext_conditioning
        self.use_contacts = use_contacts

        self.use_gender_conditioning = use_gender_conditioning
        self.num_genders = num_genders
        # Number of object classes
        self.num_classes = num_classes

        # Additional input dimensions for conditioning
        self.cond_channels = 0
        if self.use_class_conditioning:
            self.cond_channels += self.num_classes
        if self.use_pointnext_conditioning:
            self.cond_channels += 1024  # length of a feature vector

        # NEW: two subjects, each one-hot(2)    
        if self.use_gender_conditioning:
            self.cond_channels += 2 * self.num_genders  # for both sbj and second_sbj
    def get_input_with_conditioning(
        self,
        x_t: Tensor,
        t: Optional[Tensor] = None,
        t_aux: Optional[Tensor] = None,  # second timestep for unidiffuser
        sbj_gender: Optional[Tensor] = None,         # (B,) bool/int
        second_sbj_gender: Optional[Tensor] = None,  # (B,) bool/int
    ):
        # Get dimensions
        B, N = x_t.shape[:2]
        
        # Initial input is the point locations
        x_t_input = [x_t]
        x_t_cond = []

        if self.use_gender_conditioning:
            device = x_t.device

            def _to_onehot(g: Optional[Tensor]) -> Tensor:
                if g is None:
                    g = torch.randint(0, self.num_genders, (B,), device=device)
                else:
                    # in dataset bool：True=female False=male
                    # male=0 female=1
                    g = g.to(device)
                    if g.dtype == torch.bool:
                        g = g.long()
                    else:
                        g = g.long().view(-1)
                return F.one_hot(g, num_classes=self.num_genders).float()

            g1 = _to_onehot(sbj_gender)         # (B,2)
            g2 = _to_onehot(second_sbj_gender)  # (B,2)
            x_t_cond.append(torch.cat([g1, g2], dim=1))  # (B,4)

        # # dropping conditioning for regularization
        # # check train / eval flag
        # x_t_cond = torch.cat(x_t_cond, dim=1)  # (B, D_cond)
        # if self.training and torch.rand(1) < 0.1:
        #     x_t_cond = torch.zeros_like(x_t_cond)

        # Concatenate together all the features
        _input = torch.cat([*x_t_input, *x_t_cond], dim=1)  # (B, D)

        return _input
