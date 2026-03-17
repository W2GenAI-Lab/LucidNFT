import os
from typing import Iterable, Optional, Sequence

import torch
from peft import PeftModel
from torch import nn


ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def has_adapter_files(adapter_dir: str) -> bool:
    return os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")) and any(
        os.path.isfile(os.path.join(adapter_dir, filename)) for filename in ADAPTER_WEIGHT_FILES
    )


def _iter_adapter_candidates(adapter_root: str, preferred_subdirs: Sequence[str]) -> Iterable[str]:
    yielded: set[str] = set()

    def _yield(path: str) -> Iterable[str]:
        norm = os.path.abspath(path)
        if norm in yielded or not os.path.isdir(norm):
            return ()
        yielded.add(norm)
        return (norm,)

    for path in _yield(adapter_root):
        yield path

    for subdir in preferred_subdirs:
        for path in _yield(os.path.join(adapter_root, subdir)):
            yield path

    for entry in sorted(os.listdir(adapter_root)):
        path = os.path.join(adapter_root, entry)
        if has_adapter_files(path):
            for candidate in _yield(path):
                yield candidate


def resolve_peft_adapter_dir(adapter_root: str, preferred_subdirs: Sequence[str] = ()) -> str:
    if not adapter_root:
        raise ValueError("adapter_root 不能为空")
    if not os.path.isdir(adapter_root):
        raise FileNotFoundError(f"LoRA 路径不存在或不是目录: {adapter_root}")

    for candidate in _iter_adapter_candidates(adapter_root, preferred_subdirs):
        if has_adapter_files(candidate):
            return candidate

    searched = [os.path.abspath(adapter_root)]
    searched.extend(os.path.abspath(os.path.join(adapter_root, subdir)) for subdir in preferred_subdirs)
    raise FileNotFoundError(
        "未找到有效的 PEFT adapter 目录。已检查: " + ", ".join(searched)
    )


def maybe_resolve_peft_adapter_dir(adapter_root: str, preferred_subdirs: Sequence[str] = ()) -> Optional[str]:
    try:
        return resolve_peft_adapter_dir(adapter_root, preferred_subdirs)
    except (FileNotFoundError, ValueError):
        return None


def merge_peft_adapter(
    model: nn.Module,
    adapter_root: str,
    *,
    preferred_subdirs: Sequence[str] = (),
    adapter_name: str = "default",
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[nn.Module, str]:
    adapter_dir = resolve_peft_adapter_dir(adapter_root, preferred_subdirs)
    peft_model = PeftModel.from_pretrained(model, adapter_dir, adapter_name=adapter_name, is_trainable=False)
    try:
        merged_model = peft_model.merge_and_unload(safe_merge=True)
    except TypeError:
        merged_model = peft_model.merge_and_unload()

    if device is not None and dtype is not None:
        merged_model = merged_model.to(device=device, dtype=dtype)
    elif device is not None:
        merged_model = merged_model.to(device=device)
    elif dtype is not None:
        merged_model = merged_model.to(dtype=dtype)

    merged_model.eval()
    merged_model.requires_grad_(False)
    return merged_model, adapter_dir
