'''
# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""
Processor class for Qwen2-VL, extended to support audio input.
"""

import numpy as np
import torch
from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, VideoInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging

logger = logging.get_logger(__name__)


class Qwen2VLImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen2VLImagesKwargs
    _defaults = {
        "text_kwargs": {"padding": False},
    }


class Qwen2VLProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2-VL processor which wraps a Qwen2-VL image processor
    and a Qwen2 tokenizer into a single processor. Extended to handle audio.
    """
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        # vision placeholders
        self.image_token = (
            "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        )
        # audio placeholder
        self.audio_token = (
            "<|audio_pad|>" if not hasattr(tokenizer, "audio_token") else tokenizer.audio_token
        )
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio_inputs: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        # 1) Merge kwargs using the processor’s own Kwargs (with _defaults)
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # 2) Process images
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid = image_inputs["image_grid_thw"]
        else:
            image_inputs, image_grid = {}, None

        # 3) Process videos
        if videos is not None:
            video_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            video_grid = video_inputs["video_grid_thw"]
        else:
            video_inputs, video_grid = {}, None

        # 4) Prepare audio arrays & lengths
        if audio_inputs is not None:
            audio_arrays = [
                arr.cpu().numpy() if isinstance(arr, torch.Tensor) else arr for arr in audio_inputs
            ]
            audio_lengths = [arr.shape[0] for arr in audio_arrays]
        else:
            audio_arrays, audio_lengths = None, None

        # 5) Normalize text to a list
        if not isinstance(text, list):
            text = [text]

        # 6) Expand image placeholders
        if image_grid is not None:
            merge_len = self.image_processor.merge_size**2
            idx = 0
            for i, t in enumerate(text):
                while self.image_token in t:
                    reps = image_grid[idx].prod() // merge_len
                    t = t.replace(self.image_token, "<|placeholder|>" * reps, 1)
                    idx += 1
                text[i] = t.replace("<|placeholder|>", self.image_token)

        # 7) Expand video placeholders
        if video_grid is not None:
            merge_len = self.image_processor.merge_size**2
            idx = 0
            for i, t in enumerate(text):
                while self.video_token in t:
                    reps = video_grid[idx].prod() // merge_len
                    t = t.replace(self.video_token, "<|placeholder|>" * reps, 1)
                    idx += 1
                text[i] = t.replace("<|placeholder|>", self.video_token)

        # 8) Expand audio placeholders
        if audio_lengths is not None:
            idx = 0
            for i, t in enumerate(text):
                while self.audio_token in t:
                    reps = audio_lengths[idx]
                    t = t.replace(self.audio_token, "<|placeholder|>" * reps, 1)
                    idx += 1
                text[i] = t.replace("<|placeholder|>", self.audio_token)

        # 9) Tokenize (handles turning "<|audio_pad|>" into token IDs)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # 10) Concatenate raw audio arrays and attach to data dict
        data = {**text_inputs, **image_inputs, **video_inputs}
        if audio_arrays is not None:
            concat = np.concatenate(audio_arrays, axis=0)
            data["audio_concat"] = concat
            data["audio_lengths"] = np.array(audio_lengths, dtype=np.int64)

        return BatchFeature(data=data)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    @property
    def model_input_names(self):
        names = self.tokenizer.model_input_names + self.image_processor.model_input_names
        return list(dict.fromkeys(names))


__all__ = ["Qwen2VLProcessor"]
'''


# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""
Processor class for Qwen2-VL, extended to support audio input with CNN‐aware padding.
"""

# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc.
# team. All rights reserved.
#
# This code is based on EleutherAI's GPT‑NeoX library and the GPT‑NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT‑NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""
Processor class for **Qwen‑2‑VL**, extended to support **audio** input while
keeping the original file structure and naming conventions intact.

Key points (unchanged):
* Numbered comment sections match the upstream file.
* Placeholder‑token logic uses the exact `hasattr` pattern from the original
  codebase instead of `getattr`.
* Default `cnn_total_stride` restored to **4** (tiny/base/small/medium users
  can set 2 manually).
"""
#from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import torch
import whisper

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, VideoInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging

logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# util helpers (flat, as in source file)
# ---------------------------------------------------------------------------

def _ensure_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz if needed, cast to float32."""
    wav = wav.astype(np.float32)
    if sr != 16_000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16_000)
    return wav


def _waveform_to_logmel(wav: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pad/trim to 30 s and build Whisper log‑Mel spectrogram."""
    wav_t = torch.from_numpy(wav)
    wav_t = whisper.pad_or_trim(wav_t, length=30 * 16_000)
    mel = whisper.log_mel_spectrogram(wav_t)  # (80, frames)
    return mel.cpu().numpy(), mel.shape[-1]


def compute_audio_pad_count(n_mel_frames: int = None, use_qformer: bool = True, num_queries: int = 64) -> int:
    """Return the number of audio tokens based on compression method.
    
    Args:
        n_mel_frames: Number of mel frames (unused for Q-Former)
        use_qformer: If True, return fixed num_queries. If False, use conv compression.
        num_queries: Fixed number of Q-Former queries (default 64)
    
    Returns:
        Number of audio tokens to insert
    """
    if use_qformer:
        # Q-Former always outputs fixed number of tokens regardless of input length
        return num_queries
    else:
        # Legacy conv compression: Whisper CNN (2x) + conv (30x) = 60x total
        cnn_total_stride = 2
        compression_stride = 30
        return math.ceil(n_mel_frames / (cnn_total_stride * compression_stride))


# ---------------------------------------------------------------------------
# dataclass‑style kwargs
# ---------------------------------------------------------------------------
class Qwen2VLImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen2VLImagesKwargs
    _defaults = {
        "text_kwargs": {"padding": False},
    }


# ---------------------------------------------------------------------------
# main processor
# ---------------------------------------------------------------------------
class Qwen2VLProcessor(ProcessorMixin):
    r"""Vision‑Language‑Audio processor for Qwen‑2‑VL."""

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    # ---------------------------------------------------------------------
    # 0) init
    # ---------------------------------------------------------------------
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        # vision placeholders (same hasattr pattern as original code)
        self.image_token = (
            "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        )
        # audio placeholder
        self.audio_token = (
            "<|audio_pad|>" if not hasattr(tokenizer, "audio_token") else tokenizer.audio_token
        )

        # audio‑to‑token mapping parameters
        self.hop_length = 160        # 10 ms @16 kHz
        self.cnn_total_stride = 2     # ×4 down‑sampling (Whisper "large" family)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    # ------------------------------------------------------------------
    # 1) forward pass
    # ------------------------------------------------------------------
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, Sequence[TextInput], Sequence[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio_inputs: Optional[Sequence[Tuple[Union[np.ndarray, torch.Tensor], int]]] = None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        # 1) merge kwargs
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # 2) images → pixel values + grid
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid = image_inputs["image_grid_thw"]
        else:
            image_inputs, image_grid = {}, None

        # 3) videos → pixel values + grid
        if videos is not None:
            video_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            video_grid = video_inputs["video_grid_thw"]
        else:
            video_inputs, video_grid = {}, None

        # 4) audio → log‑Mel + token counts
        if audio_inputs is not None:
            mel_list: List[np.ndarray] = []
            token_counts: List[int] = []
            for wav, sr in audio_inputs:
                wav_np = wav.cpu().numpy() if isinstance(wav, torch.Tensor) else wav
                wav_np = _ensure_16k(wav_np, sr)
                mel, n_frames = _waveform_to_logmel(wav_np)
                mel_list.append(mel)
                # Use Q-Former with fixed token count
                token_counts.append(compute_audio_pad_count(n_frames, use_qformer=True, num_queries=64))
            audio_mels = np.stack(mel_list, axis=0)  # (B, 80, T)
        else:
            audio_mels, token_counts = None, None

        # 5) normalise text → list[str]
        if not isinstance(text, (list, tuple)):
            text = [text]
        text = list(text)

        # 6) expand image placeholders
        if image_grid is not None:
            merge_len = self.image_processor.merge_size ** 2
            idx = 0
            for i, t in enumerate(text):
                while self.image_token in t:
                    reps = image_grid[idx].prod() // merge_len
                    t = t.replace(self.image_token, "<|placeholder|>" * reps, 1)
                    idx += 1
                text[i] = t.replace("<|placeholder|>", self.image_token)

        # 7) expand video placeholders
        if video_grid is not None:
            merge_len = self.image_processor.merge_size ** 2
            idx = 0
            for i, t in enumerate(text):
                while self.video_token in t:
                    reps = video_grid[idx].prod() // merge_len
                    t = t.replace(self.video_token, "<|placeholder|>" * reps, 1)
                    idx += 1
                text[i] = t.replace("<|placeholder|>", self.video_token)

        # 8) expand audio placeholders
        if token_counts is not None:
            idx = 0
            for i, t in enumerate(text):
                while self.audio_token in t:
                    reps = token_counts[idx]
                    t = t.replace(self.audio_token, "<|placeholder|>" * reps, 1)
                    idx += 1
                text[i] = t.replace("<|placeholder|>", self.audio_token)

        # 9) tokenize text
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # 10) assemble final dict
        data = {**text_inputs, **image_inputs, **video_inputs}
        if audio_mels is not None:
            data["audio_mels"] = torch.from_numpy(audio_mels.astype(np.float32))
            #data["audio_lengths"] = np.asarray(token_counts, dtype=np.int64)

        return BatchFeature(data=data)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    @property
    def model_input_names(self):
        names = (
            self.tokenizer.model_input_names
            + self.image_processor.model_input_names
            + ["audio_mels", "audio_lengths"]
        )
        return list(dict.fromkeys(names))


__all__ = ["Qwen2VLProcessor"]
