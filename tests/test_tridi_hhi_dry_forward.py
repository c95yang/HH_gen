import types
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from config.model import ConditioningModelConfig, DenoisingModelConfig
from tridi.data.hh_batch_data import HHBatchData
from tridi.model.tridi import TriDiModel


class _FakeCLIPModel:
    def eval(self):
        return self

    def to(self, device):
        return self

    def encode_text(self, tokens):
        summed = tokens.float().sum(dim=1, keepdim=True)
        base = torch.arange(8, dtype=torch.float32, device=tokens.device).unsqueeze(0)
        return base + summed


def _fake_tokenize(prompts):
    vals = []
    for p in prompts:
        v = sum(ord(c) for c in str(p)) % 97
        vals.append([float(v), float((v + 11) % 97), float((v + 23) % 97)])
    return torch.tensor(vals, dtype=torch.float32)


class TestTriDiHHIDryForward(unittest.TestCase):
    def _dummy_batch(self, B=2):
        return HHBatchData(
            sbj=[f"s03_Hit_{i}" for i in range(B)],
            second_sbj=[f"s03_Hit_{i}" for i in range(B)],
            t_stamp=list(range(B)),
            interaction_label=["Hit", "Grab"],
            sbj_shape=torch.zeros(B, 10),
            sbj_global=torch.zeros(B, 6),
            sbj_pose=torch.zeros(B, 51 * 6),
            sbj_c=torch.zeros(B, 3),
            sbj_gender=torch.zeros(B, dtype=torch.bool),
            second_sbj_shape=torch.zeros(B, 10),
            second_sbj_global=torch.zeros(B, 6),
            second_sbj_pose=torch.zeros(B, 51 * 6),
            second_sbj_c=torch.zeros(B, 3),
            second_sbj_gender=torch.ones(B, dtype=torch.bool),
            scale=torch.ones(B),
        )

    def _make_model(self, use_interaction_diffusion: bool, data_interaction_channels: int = 0):
        den_cfg = DenoisingModelConfig(
            name="transformer_unidiffuser_3",
            dim_timestep_embed=32,
            params=dict(
                dim_hidden=64,
                num_layers=1,
                dim_feedforward=128,
                nhead=4,
            ),
        )
        cond_cfg = ConditioningModelConfig(
            use_gender_conditioning=True,
            num_genders=2,
            use_interaction_conditioning=use_interaction_diffusion,
            clip_model_name="ViT-B/32",
            use_interaction_contact_signature=False,
        )
        model = TriDiModel(
            data_sbj_channels=325,
            data_second_sbj_channels=325,
            data_interaction_channels=data_interaction_channels,
            use_interaction_diffusion=use_interaction_diffusion,
            denoise_mode="sample",
            beta_start=1e-5,
            beta_end=8e-3,
            beta_schedule="linear",
            denoising_model_config=OmegaConf.structured(den_cfg),
            conditioning_model_config=OmegaConf.structured(cond_cfg),
            cg_apply=False,
            cg_scale=0.0,
            cg_t_stamp=200,
        )
        return model

    def test_forward_off_keeps_two_branch_shape(self):
        batch = self._dummy_batch(B=2)
        model = self._make_model(use_interaction_diffusion=False, data_interaction_channels=0)

        denoise_loss, aux = model(batch, mode="train", return_intermediate_steps=True)
        self.assertIn("denoise_1", denoise_loss)
        self.assertIn("denoise_2", denoise_loss)
        self.assertNotIn("denoise_interaction", denoise_loss)
        self.assertEqual(tuple(aux[3].shape), (2, 650))

    def test_forward_on_adds_interaction_branch(self):
        fake_clip = types.SimpleNamespace(
            load=lambda name, device: (_FakeCLIPModel(), None),
            tokenize=_fake_tokenize,
        )
        with patch.dict("sys.modules", {"clip": fake_clip}):
            batch = self._dummy_batch(B=2)
            model = self._make_model(use_interaction_diffusion=True, data_interaction_channels=32)

            denoise_loss, aux = model(batch, mode="train", return_intermediate_steps=True)
            self.assertIn("denoise_1", denoise_loss)
            self.assertIn("denoise_2", denoise_loss)
            self.assertIn("denoise_interaction", denoise_loss)
            self.assertEqual(tuple(aux[3].shape), (2, 682))

            out = model.split_output(aux[3], aux)
            self.assertIsNotNone(out.interaction_latent)
            self.assertEqual(tuple(out.interaction_latent.shape), (2, 32))
            self.assertIsNotNone(out.timesteps_interaction)


if __name__ == "__main__":
    unittest.main()
