# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
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
"""Qwen2-VL(+Audio) model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


# -----------------------------------------------------------------------------
# Vision sub-config (unchanged)
# -----------------------------------------------------------------------------
class Qwen2VLVisionConfig(PretrainedConfig):
    model_type      = "qwen2_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int               = 32,
        embed_dim: int           = 1280,
        hidden_size: int         = 3584,
        hidden_act: str          = "quick_gelu",
        mlp_ratio: int           = 4,
        num_heads: int           = 16,
        in_channels: int         = 3,
        patch_size: int          = 14,
        spatial_merge_size: int  = 2,
        temporal_patch_size: int = 2,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth               = depth
        self.embed_dim           = embed_dim
        self.hidden_size         = hidden_size
        self.hidden_act          = hidden_act
        self.mlp_ratio           = mlp_ratio
        self.num_heads           = num_heads
        self.in_channels         = in_channels
        self.patch_size          = patch_size
        self.spatial_merge_size  = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.initializer_range   = initializer_range


# -----------------------------------------------------------------------------
# NEW Audio sub-config  (stores Whisper-style encoder hyper-params)
# -----------------------------------------------------------------------------
class Qwen2VLAudioConfig(PretrainedConfig):
    model_type      = "qwen2_vl_audio"
    base_config_key = "audio_config"

    def __init__(
        self,
        sample_rate: int              = 16_000,
        n_mels: int                   = 80,
        cnn_total_stride: int         = 2,      # stride 2×2 in Whisper convs
        max_seconds: float            = 30.0,   # pad_or_trim target
        pretrained_audio_model: str   = "openai/whisper-tiny",
        trainable: bool               = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_rate            = sample_rate
        self.n_mels                 = n_mels
        self.cnn_total_stride       = cnn_total_stride
        self.max_seconds            = max_seconds
        self.pretrained_audio_model = pretrained_audio_model
        self.trainable              = trainable


# -----------------------------------------------------------------------------
# Main config
# -----------------------------------------------------------------------------
class Qwen2VLConfig(PretrainedConfig):
    """
    Full configuration for **Qwen2-VL-Audio** models.
    Everything below is identical to the vanilla Qwen2-VL file, except where
    comments say “NEW”.
    """

    model_type = "qwen2_vl"

    # ---------------- NEW: register audio sub-config -----------------
    sub_configs = {
        "vision_config": Qwen2VLVisionConfig,
        "audio_config":  Qwen2VLAudioConfig,
    }

    # three fresh token IDs reserved for audio (<|audio_start|>, <|audio_pad|>, <|audio_end|>)
    # Pick IDs higher than any that came with the original checkpoint.
    audio_start_token_id: int = 151_657
    audio_token_id:       int = 151_658
    audio_end_token_id:   int = 151_659
    # -----------------------------------------------------------------

    keys_to_ignore_at_inference = ["past_key_values"]

    # (TP/PP plans stay exactly the same)  … ↓
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj":    "colwise",
        "layers.*.mlp.up_proj":      "colwise",
        "layers.*.mlp.down_proj":    "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers":       (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm":         (["hidden_states"], ["hidden_states"]),
    }

    # -----------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------
    def __init__(
        self,
        # -------------- text/backbone hyper-params (unchanged) -------------
        vocab_size: int           = 152_064,
        hidden_size: int          = 8192,
        intermediate_size: int    = 29_568,
        num_hidden_layers: int    = 80,
        num_attention_heads: int  = 64,
        num_key_value_heads: int  = 8,
        hidden_act: str           = "silu",
        max_position_embeddings: int = 32_768,
        initializer_range: float  = 0.02,
        rms_norm_eps: float       = 1e-5,
        use_cache: bool           = True,
        tie_word_embeddings: bool = False,
        rope_theta: float         = 1_000_000.0,
        use_sliding_window: bool  = False,
        sliding_window: int       = 4096,
        max_window_layers: int    = 80,
        attention_dropout: float  = 0.0,
        # -------------- sub-configs ---------------------------------------
        vision_config=None,
        audio_config=None,         # ← NEW
        # -------------- rope scaling (unchanged) ---------------------------
        rope_scaling=None,
        # -------------- misc kwargs ---------------------------------------
        **kwargs,
    ):
        # --------------- Vision config ------------------------------------
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        # --------------- Audio config (NEW) -------------------------------
        if isinstance(audio_config, dict):
            self.audio_config = self.sub_configs["audio_config"](**audio_config)
        elif audio_config is None:
            self.audio_config = self.sub_configs["audio_config"]()

        # --------------- backbone attrs (identical to original) -----------
        self.vocab_size             = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size            = hidden_size
        self.intermediate_size      = intermediate_size
        self.num_hidden_layers      = num_hidden_layers
        self.num_attention_heads    = num_attention_heads
        self.use_sliding_window     = use_sliding_window
        self.sliding_window         = sliding_window
        self.max_window_layers      = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads    = num_key_value_heads

        self.hidden_act             = hidden_act
        self.initializer_range      = initializer_range
        self.rms_norm_eps           = rms_norm_eps
        self.use_cache              = use_cache
        self.rope_theta             = rope_theta
        self.attention_dropout      = attention_dropout
        self.rope_scaling           = rope_scaling

        # --------------- RoPE validation (unchanged) ----------------------
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["Qwen2VLConfig", "Qwen2VLVisionConfig", "Qwen2VLAudioConfig"]
