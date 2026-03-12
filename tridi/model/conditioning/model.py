from logging import getLogger
from pathlib import Path
import hashlib
from typing import Dict, List, Optional

from diffusers import ModelMixin
import torch
import torch.nn.functional as F
from torch import Tensor

from tridi.utils.interaction import load_action_contact_signatures

logger = getLogger(__name__)


def _merge_prompts(user_prompts: Optional[Dict[str, str]]) -> Dict[str, str]:
    defaults = {
        "Grab": "Two people are interacting while one person grabs the other.",
        "Handshake": "Two people are performing a handshake.",
        "Hit": "Two people are interacting and one person hits the other.",
        "Push": "Two people are interacting and one person pushes the other.",
        "HoldingHands": "Two people are holding hands while interacting.",
        "Posing": "Two people are posing together.",
        "Hug": "Two people are hugging each other.",
        "Kick": "Two people are interacting and one person kicks the other.",
        "Unknown": "Two people are interacting.",
    }
    if user_prompts:
        defaults.update(dict(user_prompts))
    return defaults


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
        # Interaction prior
        use_interaction_conditioning: bool = False,
        interaction_source: str = "both",
        interaction_prompts: Optional[Dict[str, str]] = None,
        use_interaction_contact_signature: bool = False,
        signature_path: Optional[str] = None,
        clip_model_name: str = "ViT-B/32",
        interaction_embed_dim: Optional[int] = None,
        target_data_channels: int = 0,
        interaction_latent_channels: int = 0,
        interaction_proto_temperature: float = 0.07,
        interaction_proto_normalize: bool = True,
    ):
        super().__init__()
        # Types of conditioning
        self.use_class_conditioning = use_class_conditioning
        self.use_pointnext_conditioning = use_pointnext_conditioning
        self.use_contacts = use_contacts

        self.use_gender_conditioning = use_gender_conditioning
        self.num_genders = num_genders

        self.use_interaction_conditioning = use_interaction_conditioning
        self.interaction_source = interaction_source
        self.interaction_prompts = _merge_prompts(interaction_prompts)
        self.use_interaction_contact_signature = use_interaction_contact_signature
        self.signature_path = signature_path
        self.clip_model_name = clip_model_name
        self.interaction_embed_dim = interaction_embed_dim
        self.target_data_channels = target_data_channels
        self.interaction_latent_channels = int(interaction_latent_channels)
        self.interaction_proto_temperature = float(interaction_proto_temperature)
        self.interaction_proto_normalize = bool(interaction_proto_normalize)

        self._clip_model = None
        self._clip_tokenize = None
        self._clip_text_cache: Dict[str, Tensor] = {}
        self._interaction_state_proj_by_dim = torch.nn.ModuleDict()
        self._log_once_done = False
        self._missing_label_warned = False
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

        if self.use_interaction_conditioning and self.target_data_channels <= 0:
            raise ValueError("Interaction conditioning requires target_data_channels > 0.")

        self.action_signatures: Dict[str, Tensor] = {}
        self.signature_dim = 0
        if self.use_interaction_contact_signature:
            raw_root = None
            if self.signature_path:
                sp = Path(self.signature_path)
                raw_root = sp if sp.is_dir() else sp.parent
            else:
                for candidate in [
                    Path("data/raw/chi3d"),
                    Path("tridi/data/raw/chi3d"),
                    Path("./data/raw/chi3d"),
                ]:
                    if candidate.exists():
                        raw_root = candidate
                        break

            signatures_np = load_action_contact_signatures(
                raw_root=raw_root,
                signature_path=self.signature_path,
            )
            if len(signatures_np) > 0:
                self.signature_dim = int(next(iter(signatures_np.values())).shape[0])
                self.action_signatures = {
                    k: torch.from_numpy(v).float() for k, v in signatures_np.items()
                }

        self.interaction_projection = None
        if self.use_interaction_conditioning:
            self._ensure_clip_model_and_projection()
            # Important for checkpoint compatibility: pre-register projection modules
            # that may be referenced by checkpoints.
            if self.interaction_latent_channels > 0:
                self._ensure_interaction_state_proj(self.interaction_latent_channels)
            elif int(self.target_data_channels) > 0:
                self._ensure_interaction_state_proj(int(self.target_data_channels))

    def _ensure_clip_model_and_projection(self):
        if self._clip_model is None:
            try:
                import clip as clip_lib
            except Exception as e:
                raise ImportError(
                    "Interaction conditioning requires the `clip` package used by this repo."
                ) from e

            self._clip_model, _ = clip_lib.load(self.clip_model_name, device="cpu")
            self._clip_model = self._clip_model.eval()
            self._clip_tokenize = clip_lib.tokenize

        if self.interaction_embed_dim is None:
            with torch.no_grad():
                tokens = self._clip_tokenize(["person interaction"]).to("cpu")
                text_feat = self._clip_model.encode_text(tokens).float()
                self.interaction_embed_dim = int(text_feat.shape[-1])

        in_dim = int(self.interaction_embed_dim + self.signature_dim)
        if self.interaction_projection is None:
            self.interaction_projection = torch.nn.Sequential(
                torch.nn.Linear(in_dim, self.target_data_channels),
                torch.nn.SiLU(),
                torch.nn.Linear(self.target_data_channels, self.target_data_channels),
            )

    def _get_action_labels(self, interaction_label, batch_size: int) -> List[str]:
        if interaction_label is None:
            return ["Unknown"] * batch_size
        if isinstance(interaction_label, str):
            return [interaction_label] * batch_size
        if isinstance(interaction_label, (list, tuple)):
            return [str(x) if x is not None else "Unknown" for x in interaction_label]
        return ["Unknown"] * batch_size

    def _encode_interaction_text(self, labels: List[str], device: torch.device) -> Tensor:
        self._ensure_clip_model_and_projection()
        self._clip_model = self._clip_model.to(device)

        unique_labels = []
        for lbl in labels:
            key = str(lbl) if lbl is not None else "Unknown"
            if key not in unique_labels:
                unique_labels.append(key)
        if "Unknown" not in unique_labels:
            unique_labels.append("Unknown")

        for action in unique_labels:
            if action in self._clip_text_cache:
                continue
            prompt = self.interaction_prompts.get(action, self.interaction_prompts["Unknown"])
            with torch.no_grad():
                tokens = self._clip_tokenize([prompt]).to(device)
                feat = self._clip_model.encode_text(tokens).float()[0]
                feat = F.normalize(feat, dim=0)
            self._clip_text_cache[action] = feat.detach().cpu()

        stacked = torch.stack([
            self._clip_text_cache.get(
                action,
                self._clip_text_cache.get("Unknown")
            ) for action in labels
        ], dim=0).to(device)
        return stacked

    def _build_interaction_feature(self, labels: List[str], device: torch.device, batch_size: int) -> Tensor:
        clip_feat = self._encode_interaction_text(labels, device)
        sig_feat = self._get_signature_batch(labels, device, batch_size)
        feat = clip_feat if sig_feat is None else torch.cat([clip_feat, sig_feat], dim=1)
        return feat.to(device=device, dtype=torch.float32)

    def _ensure_interaction_state_proj(self, latent_dim: int) -> torch.nn.Module:
        self._ensure_clip_model_and_projection()
        key = str(int(latent_dim))
        if key not in self._interaction_state_proj_by_dim:
            in_dim = int(self.interaction_embed_dim + self.signature_dim)
            self._interaction_state_proj_by_dim[key] = torch.nn.Sequential(
                torch.nn.Linear(in_dim, int(latent_dim)),
                torch.nn.SiLU(),
                torch.nn.Linear(int(latent_dim), int(latent_dim)),
            )
        return self._interaction_state_proj_by_dim[key]

    def _default_candidate_actions(self) -> List[str]:
        actions = [a for a in sorted(self.interaction_prompts.keys()) if a != "Unknown"]
        return actions if len(actions) > 0 else ["Unknown"]

    def _decoder_state_signature(self, latent_dim: int, candidate_actions: List[str]) -> str:
        proj = self._ensure_interaction_state_proj(int(latent_dim))
        h = hashlib.sha256()
        for p in proj.parameters():
            t = p.detach().cpu().contiguous().numpy().tobytes()
            h.update(t)
        h.update("|".join(candidate_actions).encode("utf-8"))
        h.update(str(self.interaction_source).encode("utf-8"))
        h.update(str(bool(self.use_interaction_contact_signature)).encode("utf-8"))
        h.update(str(self.interaction_proto_temperature).encode("utf-8"))
        h.update(str(self.interaction_proto_normalize).encode("utf-8"))
        return h.hexdigest()[:16]

    def get_interaction_prototype_logits(
        self,
        interaction_latent: Tensor,
        candidate_actions: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        normalize: Optional[bool] = None,
    ) -> Dict[str, Tensor | List[str] | str | float | bool]:
        if interaction_latent is None:
            return {
                "logits": torch.empty(0),
                "similarities": torch.empty(0),
                "candidate_actions": [],
                "decoder_state_signature": "",
                "temperature": float(self.interaction_proto_temperature),
                "normalize": bool(self.interaction_proto_normalize),
            }

        if not self.use_interaction_conditioning:
            raise RuntimeError("get_interaction_prototype_logits requires use_interaction_conditioning=True")

        device = interaction_latent.device
        latent_dim = int(interaction_latent.shape[-1])
        if candidate_actions is None:
            candidate_actions = self._default_candidate_actions()

        proto_feat = self._build_interaction_feature(candidate_actions, device, len(candidate_actions))
        proj = self._ensure_interaction_state_proj(latent_dim).to(device)
        self._interaction_state_proj_by_dim = self._interaction_state_proj_by_dim.to(device)
        proto_latent = proj(proto_feat)

        use_norm = self.interaction_proto_normalize if normalize is None else bool(normalize)
        q = interaction_latent.float()
        p = proto_latent.float()
        if use_norm:
            q = F.normalize(q, dim=1)
            p = F.normalize(p, dim=1)

        similarities = q @ p.T
        temp = float(self.interaction_proto_temperature if temperature is None else temperature)
        temp = max(temp, 1e-6)
        logits = similarities / temp

        return {
            "logits": logits,
            "similarities": similarities,
            "candidate_actions": list(candidate_actions),
            "decoder_state_signature": self._decoder_state_signature(latent_dim, candidate_actions),
            "temperature": temp,
            "normalize": use_norm,
        }

    def compute_interaction_prototype_ce(
        self,
        interaction_latent: Optional[Tensor],
        interaction_label=None,
        candidate_actions: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        normalize: Optional[bool] = None,
    ) -> Dict[str, Optional[Tensor] | int | float | List[str] | str]:
        if interaction_latent is None:
            return {
                "loss": None,
                "accuracy": None,
                "num_valid": 0,
                "candidate_actions": [],
                "decoder_state_signature": "",
            }

        batch_size = int(interaction_latent.shape[0])
        labels = self._get_action_labels(interaction_label, batch_size)
        proto = self.get_interaction_prototype_logits(
            interaction_latent=interaction_latent,
            candidate_actions=candidate_actions,
            temperature=temperature,
            normalize=normalize,
        )
        candidate_actions = list(proto["candidate_actions"])
        label_to_idx = {label: idx for idx, label in enumerate(candidate_actions)}

        targets = torch.full(
            (batch_size,),
            fill_value=-100,
            dtype=torch.long,
            device=interaction_latent.device,
        )
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=interaction_latent.device)
        for i, label in enumerate(labels):
            if label in label_to_idx:
                targets[i] = int(label_to_idx[label])
                valid_mask[i] = True

        num_valid = int(valid_mask.sum().item())
        if num_valid == 0:
            return {
                "loss": None,
                "accuracy": None,
                "num_valid": 0,
                "candidate_actions": candidate_actions,
                "decoder_state_signature": proto["decoder_state_signature"],
            }

        logits = proto["logits"][valid_mask]
        target_valid = targets[valid_mask]
        if logits.device != interaction_latent.device:
            logits = logits.to(interaction_latent.device)
        if target_valid.device != interaction_latent.device:
            target_valid = target_valid.to(interaction_latent.device)
        assert logits.device == target_valid.device == interaction_latent.device, (
            f"interaction proto CE device mismatch: logits={logits.device}, "
            f"targets={target_valid.device}, latent={interaction_latent.device}"
        )
        loss = F.cross_entropy(logits, target_valid)
        pred = torch.argmax(logits, dim=1)
        acc = float((pred == target_valid).float().mean().item())

        return {
            "loss": loss,
            "accuracy": acc,
            "num_valid": num_valid,
            "candidate_actions": candidate_actions,
            "decoder_state_signature": proto["decoder_state_signature"],
        }

    def decode_interaction_latent_with_scores(
        self,
        interaction_latent: Tensor,
        topk: int = 3,
        candidate_actions: Optional[List[str]] = None,
    ) -> Dict[str, List]:
        """
        Shared decode path used by inspect/analyze scripts.
        Returns top-1/top-k plus per-class similarity scores in a stable label order.
        """
        if interaction_latent is None:
            return {
                "top1": [],
                "topk_labels": [],
                "topk_scores": [],
                "candidate_actions": [],
                "similarities": [],
                "decoder_state_signature": "",
            }

        if not self.use_interaction_conditioning:
            raise RuntimeError("decode_interaction_latent requires use_interaction_conditioning=True")

        proto = self.get_interaction_prototype_logits(
            interaction_latent=interaction_latent,
            candidate_actions=candidate_actions,
        )
        sims = proto["similarities"]
        candidate_actions = list(proto["candidate_actions"])
        latent_dim = int(interaction_latent.shape[-1])

        k = max(1, min(int(topk), int(sims.shape[1])))
        vals, idxs = torch.topk(sims, k=k, dim=1)

        top1 = [candidate_actions[int(i)] for i in idxs[:, 0].tolist()]
        topk_labels = [[candidate_actions[int(j)] for j in row] for row in idxs.tolist()]
        topk_scores = vals.tolist()

        return {
            "top1": top1,
            "topk_labels": topk_labels,
            "topk_scores": topk_scores,
            "candidate_actions": list(candidate_actions),
            "similarities": sims.tolist(),
            "decoder_state_signature": proto["decoder_state_signature"],
        }

    def get_interaction_latent(
        self,
        interaction_label=None,
        batch=None,
        device: Optional[torch.device] = None,
        latent_dim: Optional[int] = None,
    ) -> Tensor:
        if device is None:
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cpu")

        if interaction_label is None and batch is not None and hasattr(batch, "interaction_label"):
            interaction_label = batch.interaction_label

        if isinstance(interaction_label, (list, tuple)):
            batch_size = len(interaction_label)
        elif interaction_label is None:
            batch_size = 1 if batch is None else int(batch.batch_size())
        else:
            batch_size = 1

        labels = self._get_action_labels(interaction_label, batch_size)
        interaction_feat = self._build_interaction_feature(labels, device, batch_size)

        if latent_dim is None:
            latent_dim = int(interaction_feat.shape[1])

        key = str(int(latent_dim))
        self._ensure_interaction_state_proj(int(latent_dim))
        self._interaction_state_proj_by_dim = self._interaction_state_proj_by_dim.to(device)
        latent = self._interaction_state_proj_by_dim[key](interaction_feat)

        self._log_interaction_setup_once(labels)
        return latent.float()

    @torch.no_grad()
    def decode_interaction_latent(
        self,
        interaction_latent: Tensor,
        topk: int = 3,
        candidate_actions: Optional[List[str]] = None,
    ) -> Dict[str, List]:
        """
        Decode interaction latent to action labels by nearest cosine similarity
        against action prototype latents built from the same conditioning pipeline.
        Returns dict with top-1 labels and top-k labels/scores per sample.
        """
        decoded = self.decode_interaction_latent_with_scores(
            interaction_latent=interaction_latent,
            topk=topk,
            candidate_actions=candidate_actions,
        )
        return {
            "top1": decoded["top1"],
            "topk_labels": decoded["topk_labels"],
            "topk_scores": decoded["topk_scores"],
        }

    def _get_signature_batch(self, labels: List[str], device: torch.device, batch_size: int) -> Optional[Tensor]:
        if not self.use_interaction_contact_signature or self.signature_dim <= 0:
            return None

        sig = torch.zeros(batch_size, self.signature_dim, device=device, dtype=torch.float32)
        for i, action in enumerate(labels):
            if action in self.action_signatures:
                sig[i] = self.action_signatures[action].to(device=device, dtype=torch.float32)
        return sig

    def _log_interaction_setup_once(self, labels: List[str]):
        if self._log_once_done or (not self.use_interaction_conditioning):
            return
        discovered = sorted(set([x for x in labels if x and x != "Unknown"]))
        logger.info(
            "Interaction conditioning enabled: source=%s, clip_model=%s, clip_dim=%s, labels=%s",
            self.interaction_source,
            self.clip_model_name,
            self.interaction_embed_dim,
            discovered[:20],
        )
        if self.use_interaction_contact_signature:
            logger.info(
                "Interaction contact signature enabled: signature_path=%s, signature_dim=%d, known_actions=%s",
                self.signature_path,
                self.signature_dim,
                sorted(self.action_signatures.keys())[:20],
            )
        self._log_once_done = True

    def get_input_with_conditioning(
        self,
        x_t: Tensor,
        t: Optional[Tensor] = None,
        t_aux: Optional[Tensor] = None,  # second timestep for unidiffuser
        sbj_gender: Optional[Tensor] = None,         # (B,) bool/int
        second_sbj_gender: Optional[Tensor] = None,  # (B,) bool/int
        interaction_label=None,
        batch=None,
        enable_interaction_conditioning: bool = True,
    ):
        x_t = x_t.to(self.interaction_projection[0].weight.device if (self.use_interaction_conditioning and self.interaction_projection is not None) else x_t.device)
        # Get dimensions
        B, N = x_t.shape[:2]
        
        # Initial input is the point locations
        x_t_base = x_t
        x_t_cond = []

        if self.use_interaction_conditioning and enable_interaction_conditioning:
            if interaction_label is None and batch is not None and hasattr(batch, "interaction_label"):
                interaction_label = batch.interaction_label

            labels = self._get_action_labels(interaction_label, B)
            if all(lbl in [None, "", "Unknown"] for lbl in labels) and (not self._missing_label_warned):
                logger.warning(
                    "Interaction conditioning is ON but no valid interaction labels were found in batch; using 'Unknown'."
                )
                self._missing_label_warned = True

            interaction_feat = self._build_interaction_feature(labels, x_t.device, B)
            self.interaction_projection = self.interaction_projection.to(x_t.device)
            interaction_bias = self.interaction_projection(interaction_feat.to(x_t.device)).to(device=x_t.device, dtype=x_t.dtype)
            x_t_base = x_t_base + interaction_bias
            self._log_interaction_setup_once(labels)

        x_t_input = [x_t_base]

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
            x_t_cond.append(torch.cat([g1, g2], dim=1).to(device=x_t.device, dtype=x_t.dtype))  # (B,4)

        # # dropping conditioning for regularization
        # # check train / eval flag
        # x_t_cond = torch.cat(x_t_cond, dim=1)  # (B, D_cond)
        # if self.training and torch.rand(1) < 0.1:
        #     x_t_cond = torch.zeros_like(x_t_cond)

        # Concatenate together all the features
        _input = torch.cat([*x_t_input, *x_t_cond], dim=1).to(device=x_t.device, dtype=x_t.dtype)  # (B, D)

        return _input
