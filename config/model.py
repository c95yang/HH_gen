from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os


# ========================== MODEL ===========================
@dataclass
class ConditioningModelConfig:
    # class number
    use_class_conditioning: bool = False
    num_classes: int = 40  # actually number of groups - 40, 52, 64
    # pointnext encoding
    use_pointnext_conditioning: bool = False
    # contacts - for 3way unidiffuser
    use_contacts: str = "encoder_decimated_clip"  # "surface", "parts"
    contact_model: str = "gb_contacts"
    # NEW: gender conditioning
    use_gender_conditioning: bool = False
    num_genders: int = 2

    # Interaction prior conditioning (disabled by default)
    use_interaction_conditioning: bool = False
    interaction_source: str = "both"  # "seq_name" | "video_name" | "both"
    interaction_prompts: Dict[str, str] = field(default_factory=lambda: {
        "Grab": "Two people are interacting while one person grabs the other.",
        "Handshake": "Two people are performing a handshake.",
        "Hit": "Two people are interacting and one person hits the other.",
        "Push": "Two people are interacting and one person pushes the other.",
        "HoldingHands": "Two people are holding hands while interacting.",
        "Posing": "Two people are posing together.",
        "Hug": "Two people are hugging each other.",
        "Kick": "Two people are interacting and one person kicks the other.",
        "Unknown": "Two people are interacting.",
    })
    use_interaction_contact_signature: bool = False
    signature_path: Optional[str] = None
    clip_model_name: str = "ViT-B/32"
    interaction_embed_dim: Optional[int] = None
    interaction_proto_temperature: float = 0.07
    interaction_proto_normalize: bool = True

@dataclass
class DenoisingModelConfig:
    name: str = "transformer" "_" + "unidiffuser_3"  # 'simple','transformer' x "joint", "unidiffuser"
    dim_timestep_embed: int = 128
    params: Dict = field(default_factory=lambda: {})


@dataclass
class TriDiModelConfig:
    # Input configuration
    data_sbj_channels: int = 10 + 52 * 6 + 3
    data_second_sbj_channels: int = 10 + 52 * 6 + 3
    #data_obj_channels: int = 3 + 6
    #data_contact_channels: int = 128  # 256 for surface, 24 for parts
    data_interaction_channels: int = 128
    use_interaction_diffusion: bool = False

    # diffusion
    denoise_mode: str = 'sample'  # epsilon or sample (as in scheduler - prediction_type)
    beta_start: float = 1e-5  # 0.00085
    beta_end: float = 8e-3  # 0.012
    beta_schedule: str = 'linear'  # 'custom'

    # guidance for the model
    cg_apply: bool = False
    cg_scale: float = 0.0
    cg_t_stamp: int = 200
# ============================================================