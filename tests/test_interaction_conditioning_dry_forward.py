import types
import unittest
from unittest.mock import patch
import importlib.util
from pathlib import Path
import sys

import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tridi" / "model" / "conditioning" / "model.py"


def _load_conditioning_module():
    spec = importlib.util.spec_from_file_location("interaction_conditioning_model", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    with patch.dict(sys.modules, {"diffusers": types.SimpleNamespace(ModelMixin=torch.nn.Module)}):
        spec.loader.exec_module(mod)
    return mod


class _FakeCLIPModel:
    def eval(self):
        return self

    def to(self, device):
        return self

    def encode_text(self, tokens):
        # Deterministic fake features based on token ids.
        summed = tokens.float().sum(dim=1, keepdim=True)
        base = torch.arange(8, dtype=torch.float32, device=tokens.device).unsqueeze(0)
        return base + summed


def _fake_tokenize(prompts):
    vals = []
    for p in prompts:
        v = sum(ord(c) for c in str(p)) % 97
        vals.append([float(v), float((v + 13) % 97), float((v + 29) % 97)])
    return torch.tensor(vals, dtype=torch.float32)


class TestInteractionConditioningDryForward(unittest.TestCase):
    def test_disabled_no_clip_call_and_shape_unchanged(self):
        _MOD = _load_conditioning_module()
        ConditioningModel = _MOD.ConditioningModel
        x_t = torch.randn(2, 650)

        def _fail_clip_load(*args, **kwargs):
            raise AssertionError("clip.load should not be called when interaction conditioning is disabled")

        fake_clip_module = types.SimpleNamespace(
            load=_fail_clip_load,
            tokenize=_fake_tokenize,
        )

        with patch.dict("sys.modules", {"clip": fake_clip_module}):
            with patch.object(_MOD, "logger"):
                cm = ConditioningModel(
                    use_gender_conditioning=False,
                    use_interaction_conditioning=False,
                    target_data_channels=650,
                )
                out = cm.get_input_with_conditioning(x_t)

        self.assertEqual(tuple(out.shape), tuple(x_t.shape))

    def test_enabled_shape_unchanged_and_action_dependent(self):
        _MOD = _load_conditioning_module()
        ConditioningModel = _MOD.ConditioningModel
        fake_clip_module = types.SimpleNamespace(
            load=lambda name, device: (_FakeCLIPModel(), None),
            tokenize=_fake_tokenize,
        )

        with patch.dict("sys.modules", {"clip": fake_clip_module}):
            cm = ConditioningModel(
                use_gender_conditioning=False,
                use_interaction_conditioning=True,
                interaction_prompts={
                    "Grab": "grab interaction",
                    "Hit": "hit interaction",
                    "Unknown": "unknown interaction",
                },
                clip_model_name="ViT-B/32",
                target_data_channels=650,
            )

            x_t = torch.zeros(2, 650)
            out = cm.get_input_with_conditioning(x_t, interaction_label=["Grab", "Hit"])

            self.assertEqual(tuple(out.shape), tuple(x_t.shape))
            self.assertFalse(torch.allclose(out[0], out[1]))
            self.assertGreater(torch.norm(out[0] - out[1]).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
