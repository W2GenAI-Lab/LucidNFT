import argparse
import os

import numpy as np
import torch
from einops import rearrange
from torchvision.utils import save_image

from src.flux.align_color import wavelet_reconstruction
from src.flux.flux_prior_redux_ir import siglip_from_unit_tensor
from src.flux.lucidflux import (
    load_checkpoint_bundle,
    load_precomputed_embeddings,
    load_connector,
    load_siglip_model,
    load_swinir,
    move_modules_to_device,
    prepare_with_embeddings,
    preprocess_lq_image,
)
from src.flux.sampling import denoise_lucidflux, get_noise, get_schedule, unpack


DEFAULT_SWINIR_PRETRAINED = "weights/swinir.pth"
DEFAULT_LUCIDFLUX_CHECKPOINT = "weights/lucidflux/lucidflux.pth"
DEFAULT_LUCIDFLUX_PROMPT_EMBEDDINGS = "weights/lucidflux/prompt_embeddings.pt"
DEFAULT_LUCIDFLUX_SIGLIP_CKPT = "weights/siglip"


def create_argparser():
    parser = argparse.ArgumentParser(
        description="Inference entrypoint for LucidFlux and optional LucidFlux+LucidNFT LoRA adapters."
    )
    parser.add_argument("--control_image", type=str, required=True, help="Input image path or directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g. cpu, cuda:0)")
    parser.add_argument("--offload", action="store_true", help="Offload modules to CPU between stages")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument("--height", type=int, default=1024, help="Output height")
    parser.add_argument("--num_steps", type=int, default=24, help="Number of diffusion steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=123456789, help="Random seed")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16", help="Inference dtype for the diffusion/control path")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_LUCIDFLUX_CHECKPOINT, help="LucidFlux checkpoint path")
    parser.add_argument("--prompt_embeddings", type=str, default=DEFAULT_LUCIDFLUX_PROMPT_EMBEDDINGS, help="Precomputed prompt embeddings path")
    parser.add_argument("--swinir_pretrained", type=str, default=DEFAULT_SWINIR_PRETRAINED, help="SwinIR checkpoint path")
    parser.add_argument("--siglip_ckpt", type=str, default=DEFAULT_LUCIDFLUX_SIGLIP_CKPT, help="SigLIP checkpoint path")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LucidNFT LoRA root or adapter directory")
    return parser


def collect_input_paths(control_image: str):
    if os.path.isdir(control_image):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
        input_paths = [
            os.path.join(control_image, filename)
            for filename in sorted(os.listdir(control_image))
            if os.path.isfile(os.path.join(control_image, filename)) and filename.lower().endswith(exts)
        ]
        if not input_paths:
            raise ValueError(f"No image files found in directory: {control_image}")
        return input_paths
    return [control_image]


def main(args):
    name = "flux-dev"
    is_schnell = name == "flux-schnell"
    torch_device = torch.device(args.device)
    model_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    encoder_dtype = model_dtype if torch_device.type == "cuda" else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    embeddings = load_precomputed_embeddings(args.prompt_embeddings, torch_device)
    print(f"Loading precomputed embeddings from {args.prompt_embeddings}")
    print(f"Loaded embeddings for prompt: '{embeddings['prompt']}'")
    print(f"txt shape: {embeddings['txt'].shape}, vec shape: {embeddings['vec'].shape}")

    model, ae, dual_condition_branch, connector_source = load_checkpoint_bundle(
        name,
        args.checkpoint,
        args.lora_path,
        torch_device,
        model_dtype,
        args.offload,
    )

    swinir = load_swinir(torch_device, args.swinir_pretrained, args.offload)
    siglip_model = load_siglip_model(args.siglip_ckpt, torch_device, encoder_dtype, args.offload)
    connector = load_connector("cpu" if args.offload else torch_device, encoder_dtype, connector_source)

    width = 16 * args.width // 16
    height = 16 * args.height // 16
    timesteps = get_schedule(
        args.num_steps,
        (width // 8) * (height // 8) // (16 * 16),
        shift=(not is_schnell),
    )

    for img_path in collect_input_paths(args.control_image):
        filename = os.path.basename(img_path).split(".")[0]
        lq_processed = preprocess_lq_image(img_path, width, height)
        condition_cond = torch.from_numpy((np.array(lq_processed) / 127.5) - 1)
        condition_cond = condition_cond.permute(2, 0, 1).unsqueeze(0).to(model_dtype).to(torch_device)

        with torch.no_grad():
            ci_01 = torch.clamp((condition_cond.float() + 1.0) / 2.0, 0.0, 1.0)
            if args.offload:
                swinir.to(torch_device)
            ci_pre = swinir(ci_01.to(torch_device)).float().clamp(0.0, 1.0)
            if args.offload:
                swinir.to("cpu")
            condition_cond_ldr = (ci_pre * 2.0 - 1.0).to(model_dtype)

            torch.manual_seed(args.seed)
            x = get_noise(1, height, width, device=torch_device, dtype=model_dtype, seed=args.seed)
            inp_cond = prepare_with_embeddings(
                img=x,
                precomputed_txt=embeddings["txt"],
                precomputed_vec=embeddings["vec"],
            )

            siglip_size = getattr(getattr(siglip_model, "config", None), "image_size", 512)
            siglip_pixel_values_pre = siglip_from_unit_tensor(ci_pre, size=(siglip_size, siglip_size))
            inputs = {"pixel_values": siglip_pixel_values_pre.to(device=torch_device, dtype=encoder_dtype)}
            if args.offload:
                siglip_model.to(torch_device)
            siglip_image_pre_fts = siglip_model(**inputs).last_hidden_state.to(dtype=encoder_dtype)
            if args.offload:
                siglip_model.to("cpu")
                torch.cuda.empty_cache()

            connector_dtype = connector.redux_up.weight.dtype
            if args.offload:
                connector.to(torch_device)
            image_embeds = connector(
                siglip_image_pre_fts.to(device=torch_device, dtype=connector_dtype)
            )["image_embeds"]
            if args.offload:
                connector.to("cpu")
                torch.cuda.empty_cache()

            txt = inp_cond["txt"].to(device=torch_device, dtype=model_dtype)
            txt_ids = inp_cond["txt_ids"].to(device=torch_device, dtype=model_dtype)
            siglip_txt = torch.cat([txt, image_embeds.to(dtype=model_dtype)], dim=1)
            batch_size, _, channels = txt_ids.shape
            extra_ids = torch.zeros((batch_size, 1024, channels), device=txt_ids.device, dtype=model_dtype)
            siglip_txt_ids = torch.cat([txt_ids, extra_ids], dim=1).to(dtype=model_dtype)

            if args.offload:
                move_modules_to_device(torch_device, model, dual_condition_branch)
                torch.cuda.empty_cache()

            x = denoise_lucidflux(
                model,
                dual_condition_model=dual_condition_branch,
                img=inp_cond["img"],
                img_ids=inp_cond["img_ids"],
                txt=txt,
                txt_ids=txt_ids,
                siglip_txt=siglip_txt,
                siglip_txt_ids=siglip_txt_ids,
                vec=inp_cond["vec"],
                timesteps=timesteps,
                guidance=args.guidance,
                condition_cond_lq=condition_cond,
                condition_cond_ldr=condition_cond_ldr,
            )

            if args.offload:
                move_modules_to_device("cpu", model, dual_condition_branch)
                torch.cuda.empty_cache()
                ae.decoder.to(x.device)

            x = unpack(x.float(), height, width)
            x = ae.decode(x)
            if args.offload:
                ae.decoder.cpu()
                torch.cuda.empty_cache()

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        hq = wavelet_reconstruction((x1.permute(2, 0, 1) + 1.0) / 2, ci_pre.squeeze(0))
        hq = hq.clamp(0, 1)
        save_image(hq, os.path.join(args.output_dir, f"{filename}.png"))
        print(f"[INFO] {filename} is done. Path: {args.output_dir}")


if __name__ == "__main__":
    main(create_argparser().parse_args())
