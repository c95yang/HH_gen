import pickle as pkl

import clip
import numpy as np
import torch

from tridi.model.conditioning.contact_ae_clip import ContactEnc, ContactEncCLIP, ContactDec
from tridi.utils.contacts import contact_to_caption


class TextConditionModel:
    """
    A class that wraps contact encoding and decoding models.
    """
    def __init__(self, model_type, device='cpu', weights_path = None) -> None:
        self.model_type = model_type

        self.clip_model = None
        self.text_condition_ae = None
        self.enc = None
        self.dec = None

        self.device = device

        if model_type == "NONE":
            return None
        
        if model_type == "clip":
            text_condition_ae = ContactEncCLIP(512, 256, 128)
            self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
            self.clip_model = self.clip_model.eval()

        text_condition_ae = text_condition_ae.to("cuda")

        if weights_path is not None:
            checkpoint = torch.load(weights_path, map_location='cuda', weights_only=False)
            state_dict = checkpoint[model_type]
            missing_keys, unexpected_keys = text_condition_ae.load_state_dict(state_dict, strict=True)

        text_condition_ae = torch.compile(text_condition_ae, mode="reduce-overhead")
        text_condition_ae = text_condition_ae.eval()

        self.dec = text_condition_ae
        self.enc = text_condition_ae
