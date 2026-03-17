"""Adapted from Qwen's qwen3_vl_embedding.py script:
https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/blob/main/scripts/qwen3_vl_embedding.py
"""

import logging
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers.modeling_outputs import ModelOutput

try:
    from qwen_vl_utils.vision_process import process_vision_info
except ImportError as exc:
    process_vision_info = None
    _QWEN_VL_UTILS_IMPORT_ERROR = exc
else:
    _QWEN_VL_UTILS_IMPORT_ERROR = None

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
except ImportError as exc:
    Qwen3VLConfig = None
    Qwen3VLModel = None
    Qwen3VLPreTrainedModel = nn.Module
    Qwen3VLProcessor = None
    _QWEN3_IMPORT_ERROR = exc
else:
    _QWEN3_IMPORT_ERROR = None


logger = logging.getLogger(__name__)

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS


def _ensure_qwen_dependencies() -> None:
    if _QWEN3_IMPORT_ERROR is not None:
        raise ImportError(
            "Qwen3-VL support is unavailable. Install a transformers version that provides "
            "`transformers.models.qwen3_vl` before using LucidConsistency."
        ) from _QWEN3_IMPORT_ERROR
    if _QWEN_VL_UTILS_IMPORT_ERROR is not None:
        raise ImportError(
            "LucidConsistency requires `qwen-vl-utils` for vision preprocessing."
        ) from _QWEN_VL_UTILS_IMPORT_ERROR


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config_class = Qwen3VLConfig
    config: "Qwen3VLConfig"

    def __init__(self, config):
        _ensure_qwen_dependencies()
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        del logits_to_keep
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def sample_frames(
    frames: List[Union[str, Image.Image]],
    num_segments: int,
    max_segments: int,
) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    frame_ids = np.linspace(0, duration - 1, num_segments, dtype=int).tolist()
    sampled_frames: List[Union[str, Image.Image]] = []
    last_frame = frames[frame_ids[-1]]
    for frame_idx in frame_ids:
        try:
            sampled_frames.append(frames[frame_idx])
        except Exception:
            break
    while len(sampled_frames) < num_segments:
        sampled_frames.append(last_frame)
    return sampled_frames[:max_segments]


class Qwen3VLEmbedder:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        num_frames: int = MAX_FRAMES,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        train_backbone: bool = False,
        **kwargs,
    ):
        _ensure_qwen_dependencies()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **kwargs,
        ).to(device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path,
            padding_side="right",
        )

        self.device = next(self.model.parameters()).device
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            self.embedding_dim = text_cfg.hidden_size
        else:
            self.embedding_dim = self.model.config.vision_config.out_hidden_size
        self.proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        ).to(self.device)

        self._set_trainable(train_backbone)

    def _set_trainable(self, train_backbone: bool) -> None:
        self.train_backbone = train_backbone
        for param in self.model.parameters():
            param.requires_grad = train_backbone
        if train_backbone:
            self.model.train()
        else:
            self.model.eval()
        self.proj.train()

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.train_backbone:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attention_mask": inputs.get("attention_mask"),
        }

    def format_model_input(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image]] = None,
        video: Optional[Union[str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
                instruction = instruction + "."

        content: List[Dict[str, Any]] = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content},
        ]

        if not text and not image and not video:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        if video:
            video_content = None
            video_kwargs: Dict[str, Any] = {"total_pixels": self.total_pixels}
            if isinstance(video, list):
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    video_content = sample_frames(video_content, self.num_frames, self.max_frames)
                video_content = [("file://" + item if isinstance(item, str) else item) for item in video_content]
            elif isinstance(video, str):
                video_content = video if video.startswith(("http://", "https://")) else "file://" + video
                video_kwargs = {"fps": fps or self.fps, "max_frames": max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(video)}")

            if video_content:
                content.append({"type": "video", "video": video_content, **video_kwargs})

        if image:
            if isinstance(image, Image.Image):
                image_content: Union[str, Image.Image] = image
            elif isinstance(image, str):
                image_content = image if image.startswith(("http", "oss")) else "file://" + image
            else:
                raise TypeError(f"Unrecognized image type: {type(image)}")
            content.append(
                {
                    "type": "image",
                    "image": image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )

        if text:
            content.append({"type": "text", "text": text})

        return conversation

    def _preprocess_inputs(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except Exception as exc:
            logger.error("Error in processing vision info: %s", exc)
            images = None
            video_inputs = None
            video_kwargs = {"do_sample_frames": False}
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": "NULL"}]}],
                add_generation_prompt=True,
                tokenize=False,
            )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        return self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )

    @staticmethod
    def _pool_last_token(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        reversed_mask = attention_mask.flip(dims=[1])
        last_valid = reversed_mask.argmax(dim=1)
        col = attention_mask.shape[1] - last_valid - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def process(
        self,
        inputs: List[Dict[str, Any]],
        normalize: bool = True,
        return_backbone: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        conversations = [
            self.format_model_input(
                text=item.get("text"),
                image=item.get("image"),
                video=item.get("video"),
                instruction=item.get("instruction"),
                fps=item.get("fps"),
                max_frames=item.get("max_frames"),
            )
            for item in inputs
        ]
        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {key: value.to(self.device) for key, value in processed_inputs.items()}

        outputs = self.forward(processed_inputs)
        backbone_embeddings = self._pool_last_token(
            outputs["last_hidden_state"],
            outputs["attention_mask"],
        )

        proj_dtype = self.proj[0].weight.dtype
        if backbone_embeddings.dtype != proj_dtype:
            backbone_embeddings = backbone_embeddings.to(proj_dtype)

        proj_embeddings = self.proj(backbone_embeddings)
        if normalize:
            backbone_embeddings = F.normalize(backbone_embeddings, p=2, dim=-1)
            proj_embeddings = F.normalize(proj_embeddings, p=2, dim=-1)
        if return_backbone:
            return backbone_embeddings, proj_embeddings
        return proj_embeddings

    def encode_image(
        self,
        image: Union[Image.Image, str, List[Union[Image.Image, str]]],
        instruction: Optional[str] = None,
        normalize: bool = True,
        return_backbone: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(image, list):
            inputs = [{"image": item, "instruction": instruction} for item in image]
        else:
            inputs = [{"image": image, "instruction": instruction}]
        return self.process(inputs, normalize=normalize, return_backbone=return_backbone)

    def load_proj_head(self, state_dict_path: str, strict: bool = True) -> None:
        state = torch.load(state_dict_path, map_location=self.device)
        self.proj.load_state_dict(state, strict=strict)


__all__ = [
    "Qwen3VLForEmbedding",
    "Qwen3VLForEmbeddingOutput",
    "Qwen3VLEmbedder",
    "sample_frames",
]
