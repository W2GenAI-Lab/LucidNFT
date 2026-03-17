import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from PIL import Image

from .peft_utils import maybe_resolve_peft_adapter_dir, merge_peft_adapter
from .swinir import SwinIR
from .util import load_ae, load_flow_model, load_safetensors, load_single_condition_branch


TRANSFORMER_LORA_SUBDIRS = ("lora_dit",)
CONDITION_LORA_SUBDIRS = ("lora_condition",)


def _expand_batch(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return repeat(tensor, "1 ... -> b ...", b=batch_size)
    raise ValueError(f"{name} batch size {tensor.shape[0]} does not match expected batch size {batch_size}")


def move_modules_to_device(device: torch.device | str, *modules: nn.Module) -> None:
    for module in modules:
        module.to(device)


def prepare_with_embeddings(
    img: torch.Tensor,
    precomputed_txt: torch.Tensor,
    precomputed_vec: torch.Tensor,
):
    bs, _, h, w = img.shape

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device, dtype=img.dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt = _expand_batch(precomputed_txt, bs, "precomputed_txt").to(img.device)
    vec = _expand_batch(precomputed_vec, bs, "precomputed_vec").to(img.device)
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=img.device, dtype=img.dtype)

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False) if cond_proj_dim is not None else None
        self.act = get_activation(act_fn)
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)
        self.post_act = get_activation(post_act_fn) if post_act_fn is not None else None

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Modulation(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)
        self.control_index_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=dim)

    def forward(self, x, timestep, control_index):
        timesteps_proj = self.time_proj(timestep * 1000)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=x.dtype))

        if control_index.dim() == 0:
            control_index = control_index.repeat(x.shape[0])
        elif control_index.dim() == 1 and control_index.shape[0] != x.shape[0]:
            control_index = control_index.expand(x.shape[0])

        control_index = control_index.to(device=x.device, dtype=x.dtype)
        control_index_proj = self.time_proj(control_index)
        control_index_emb = self.control_index_embedder(control_index_proj.to(dtype=x.dtype))
        timesteps_emb = timesteps_emb + control_index_emb
        emb = self.linear(self.silu(timesteps_emb))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]


class DualConditionComposer(nn.Module):
    def __init__(
        self,
        condition_branch_lq: nn.Module,
        condition_branch_ldr: nn.Module,
        modulation_lq: nn.Module,
        modulation_ldr: nn.Module,
    ):
        super().__init__()
        self.lq = condition_branch_lq
        self.pre = condition_branch_ldr
        self.modulation_lq = modulation_lq
        self.modulation_pre = modulation_ldr

    def forward(
        self,
        *,
        img,
        img_ids,
        condition_cond_lq,
        txt,
        txt_ids,
        y,
        timesteps,
        guidance,
        condition_cond_ldr=None,
    ):
        out_lq = self.lq(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_lq,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )

        out_ldr = self.pre(
            img=img,
            img_ids=img_ids,
            controlnet_cond=condition_cond_ldr,
            txt=txt,
            txt_ids=txt_ids,
            y=y,
            timesteps=timesteps,
            guidance=guidance,
        )
        out = []
        num_blocks = 19
        for i in range(num_blocks // 2 + 1):
            for control_index, (lq, ldr) in enumerate(zip(out_lq, out_ldr)):
                control_index_tensor = torch.tensor(control_index, device=timesteps.device, dtype=timesteps.dtype)
                lq = self.modulation_lq(lq, timesteps, i * 2 + control_index_tensor)

                if len(out) == num_blocks:
                    break

                ldr = self.modulation_pre(ldr, timesteps, i * 2 + control_index_tensor)
                out.append(lq + ldr)
        return out


class DualConditionBranch(nn.Module):
    def __init__(self, dual_condition_branch: nn.Module, connector: nn.Module):
        super().__init__()
        self.dual_condition_branch = dual_condition_branch
        self.connector = connector

    def forward(self, *args, **kwargs):
        return self.dual_condition_branch(*args, **kwargs)


def preprocess_lq_image(image_path: str, width: int, height: int):
    image = Image.open(image_path).convert("RGB")
    return image.resize((width, height))


def load_state_dict_any(path: str):
    if path.endswith(".safetensors"):
        return load_safetensors(path)
    state = torch.load(path, map_location="cpu")
    return state.get("state_dict", state) if isinstance(state, dict) else state


def load_lucidflux_weights(checkpoint_path: str):
    return load_state_dict_any(checkpoint_path)


def load_precomputed_embeddings(embeddings_path: str, device: torch.device | str):
    embeddings_data = torch.load(embeddings_path, map_location="cpu")
    return {
        "txt": embeddings_data["txt"].to(device),
        "vec": embeddings_data["vec"].to(device),
        "prompt": embeddings_data.get("prompt", "Unknown prompt"),
    }


def load_dual_condition_branch(
    name: str,
    checkpoint: dict,
    device: torch.device | str,
    offload: bool,
    branch_dtype: torch.dtype = torch.bfloat16,
    modulation_dim: int = 3072,
):
    load_device = "cpu" if offload else device
    target_device = "cpu" if offload else device

    condition_lq = load_single_condition_branch(name, load_device).to(branch_dtype)
    condition_lq.load_state_dict(checkpoint["condition_lq"], strict=False)
    condition_lq = condition_lq.to(device)

    condition_ldr = load_single_condition_branch(name, load_device).to(branch_dtype)
    condition_ldr.load_state_dict(checkpoint["condition_ldr"], strict=False)

    modulation_lq = Modulation(dim=modulation_dim).to(branch_dtype)
    modulation_lq.load_state_dict(checkpoint["modulation_lq"], strict=False)

    modulation_ldr = Modulation(dim=modulation_dim).to(branch_dtype)
    modulation_ldr.load_state_dict(checkpoint["modulation_ldr"], strict=False)

    return DualConditionComposer(
        condition_lq,
        condition_ldr,
        modulation_lq=modulation_lq,
        modulation_ldr=modulation_ldr,
    ).to(target_device)


def load_swinir(device: torch.device | str, checkpoint_path: str, offload: bool):
    swinir = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        sf=8,
        img_range=1.0,
        upsampler="nearest+conv",
        resi_connection="1conv",
        unshuffle=True,
        unshuffle_scale=8,
    )
    ckpt_obj = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt_obj.get("state_dict", ckpt_obj)
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    swinir.load_state_dict(new_state, strict=False)
    swinir.eval()
    for param in swinir.parameters():
        param.requires_grad_(False)
    return swinir.to("cpu" if offload else device)


def load_siglip_model(siglip_ckpt: str, device: torch.device | str, dtype: torch.dtype, offload: bool):
    from transformers import SiglipVisionModel

    siglip_model = SiglipVisionModel.from_pretrained(siglip_ckpt)
    siglip_model.eval()
    siglip_model.to("cpu" if offload else device).to(dtype=dtype)
    return siglip_model


def load_connector(device: torch.device | str, dtype: torch.dtype, source):
    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder

    connector = ReduxImageEncoder()
    state_dict = load_state_dict_any(source) if isinstance(source, str) else source
    connector.load_state_dict(state_dict, strict=False)
    connector.eval()
    connector.to(device).to(dtype=dtype)
    return connector


def load_checkpoint_bundle(
    name: str,
    checkpoint_path: str,
    lora_path: str | None,
    torch_device: torch.device,
    model_dtype: torch.dtype,
    offload: bool,
):
    model = load_flow_model(name, device=torch_device).to(dtype=model_dtype)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    checkpoint = load_lucidflux_weights(checkpoint_path)
    dual_condition_branch = load_dual_condition_branch(
        name,
        checkpoint,
        torch_device,
        offload=offload,
        branch_dtype=model_dtype,
    )

    if lora_path:
        model, transformer_lora_dir = merge_peft_adapter(
            model,
            lora_path,
            preferred_subdirs=TRANSFORMER_LORA_SUBDIRS,
            adapter_name="inference_transformer",
            device=torch_device,
            dtype=next(model.parameters()).dtype,
        )
        print(f"Merged transformer LoRA from {transformer_lora_dir}")

        condition_lora_dir = maybe_resolve_peft_adapter_dir(lora_path, CONDITION_LORA_SUBDIRS)
        if condition_lora_dir and condition_lora_dir != transformer_lora_dir:
            condition_device = "cpu" if offload else torch_device
            condition_dtype = next(dual_condition_branch.parameters()).dtype
            dual_condition_branch_connector = DualConditionBranch(
                dual_condition_branch=dual_condition_branch,
                connector=nn.Identity(),
            ).to(condition_device)
            dual_condition_branch_connector, _ = merge_peft_adapter(
                dual_condition_branch_connector,
                condition_lora_dir,
                adapter_name="inference_condition",
                device=condition_device,
                dtype=condition_dtype,
            )
            dual_condition_branch = dual_condition_branch_connector.dual_condition_branch
            print(f"Merged condition LoRA from {condition_lora_dir}")

    return model, ae, dual_condition_branch, checkpoint["connector"]


__all__ = [
    "DualConditionBranch",
    "DualConditionComposer",
    "Modulation",
    "Timesteps",
    "TimestepEmbedding",
    "_expand_batch",
    "get_activation",
    "get_timestep_embedding",
    "load_checkpoint_bundle",
    "load_dual_condition_branch",
    "load_lucidflux_weights",
    "load_precomputed_embeddings",
    "load_connector",
    "load_siglip_model",
    "load_state_dict_any",
    "load_swinir",
    "move_modules_to_device",
    "prepare_with_embeddings",
    "preprocess_lq_image",
]
