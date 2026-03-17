import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

from huggingface_hub import hf_hub_download, snapshot_download


FLUX_REPO = "black-forest-labs/FLUX.1-dev"
FLOW_FILE = "flux1-dev.safetensors"
AE_FILE = "ae.safetensors"
SWINIR_REPO = "lxq007/DiffBIR"
SWINIR_FILE = "general_swinir_v1.ckpt"
LUCIDFLUX_REPO = "W2GenAI/LucidFlux"
LUCIDFLUX_FILE = "lucidflux.pth"
PROMPT_EMBEDDINGS_FILE = "prompt_embeddings.pt"
ULTRAFLUX_REPO = "Owen777/UltraFlux-v1"
SIGLIP_REPO = "google/siglip2-so400m-patch16-512"
LUCIDNFT_REPO = "W2GenAI/LucidNFT"
LUCIDNFT_LORA_DIRNAME = "LucidFlux+LucidNFT_lora"
LUCIDNFT_TRANSFORMER_SUBDIR = "lora_dit"
LUCIDNFT_CONDITION_SUBDIR = "lora_condition"
LUCIDCONSISTENCY_DIRNAME = "LucidConsistency"
LUCIDCONSISTENCY_PROJ_HEAD_FILE = "proj_head.pt"
QWEN_EMBEDDING_REPO = "Qwen/Qwen3-VL-Embedding-8B"
QWEN_EMBEDDING_DIRNAME = "Qwen3-VL-Embedding-8B"
MODEL_KEY = "flux-dev"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download fixed LucidNFT inference weights")
    parser.add_argument("--dest", type=str, default="weights", help="Destination root directory")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--print-env", action="store_true", help="Also print export lines to stdout")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def plan(dest_root: Path) -> Tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    model_dir = dest_root / MODEL_KEY
    flow_dst = model_dir / FLOW_FILE
    ae_dst = model_dir / AE_FILE
    env_path = dest_root / "env.sh"
    manifest_path = dest_root / "manifest.json"
    swinir_dst = dest_root / "swinir.pth"
    lucidflux_dst = dest_root / "lucidflux" / LUCIDFLUX_FILE
    prompt_embeddings_dst = dest_root / "lucidflux" / PROMPT_EMBEDDINGS_FILE
    lucidnft_lora_dir = dest_root / "lucidflux" / LUCIDNFT_LORA_DIRNAME
    ultraflux_dir = dest_root / "ultraflux"
    siglip_dir = dest_root / "siglip"
    lucidconsistency_dir = dest_root / LUCIDCONSISTENCY_DIRNAME
    proj_head_dst = lucidconsistency_dir / LUCIDCONSISTENCY_PROJ_HEAD_FILE
    qwen_embedding_dir = lucidconsistency_dir / QWEN_EMBEDDING_DIRNAME
    return (
        flow_dst,
        ae_dst,
        env_path,
        manifest_path,
        swinir_dst,
        lucidflux_dst,
        prompt_embeddings_dst,
        lucidnft_lora_dir,
        ultraflux_dir,
        siglip_dir,
        proj_head_dst,
        qwen_embedding_dir,
    )


def env_lines(flow_dst: Path, ae_dst: Path) -> Tuple[str, str]:
    prefix = MODEL_KEY.replace("-", "_").upper()
    return (
        f"export {prefix}_FLOW={flow_dst}",
        f"export {prefix}_AE={ae_dst}",
    )


def write_env(env_path: Path, flow_dst: Path, ae_dst: Path) -> None:
    l1, l2 = env_lines(flow_dst, ae_dst)
    content = "\n".join([l1, l2, "", f"# source {env_path}"]) + "\n"
    env_path.write_text(content)


def write_manifest(path: Path, data: Dict[str, str]) -> None:
    import json

    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    dest_root = Path(args.dest).resolve()

    (
        flow_dst,
        ae_dst,
        env_path,
        manifest_path,
        swinir_dst,
        lucidflux_dst,
        prompt_embeddings_dst,
        lucidnft_lora_dir,
        ultraflux_dir,
        siglip_dir,
        proj_head_dst,
        qwen_embedding_dir,
    ) = plan(dest_root)

    if args.dry_run:
        l1, l2 = env_lines(flow_dst, ae_dst)
        sys.stdout.write(
            "\n".join(
                [
                    f"DRY RUN: download FLOW {FLUX_REPO}:{FLOW_FILE} -> {flow_dst}",
                    f"DRY RUN: download AE   {FLUX_REPO}:{AE_FILE} -> {ae_dst}",
                    f"DRY RUN: download SwinIR {SWINIR_REPO}:{SWINIR_FILE} -> {swinir_dst}",
                    f"DRY RUN: download LucidFlux {LUCIDFLUX_REPO}:{LUCIDFLUX_FILE} -> {lucidflux_dst}",
                    f"DRY RUN: download Prompt Embeddings {LUCIDFLUX_REPO}:{PROMPT_EMBEDDINGS_FILE} -> {prompt_embeddings_dst}",
                    (
                        f"DRY RUN: snapshot LucidNFT LoRAs {LUCIDNFT_REPO}:"
                        f"{LUCIDNFT_TRANSFORMER_SUBDIR}/*,{LUCIDNFT_CONDITION_SUBDIR}/* -> {lucidnft_lora_dir}"
                    ),
                    f"DRY RUN: snapshot UltraFlux VAE {ULTRAFLUX_REPO}:vae/* -> {ultraflux_dir}",
                    f"DRY RUN: snapshot SIGLIP {SIGLIP_REPO} -> {siglip_dir}",
                    f"DRY RUN: download LucidConsistency proj head {LUCIDNFT_REPO}:{LUCIDCONSISTENCY_PROJ_HEAD_FILE} -> {proj_head_dst}",
                    f"DRY RUN: snapshot Qwen embedding model {QWEN_EMBEDDING_REPO} -> {qwen_embedding_dir}",
                    "DRY RUN: write env exports",
                    l1,
                    l2,
                ]
            )
            + "\n"
        )
        return 0

    ensure_dir(dest_root)
    ensure_dir(flow_dst.parent)
    ensure_dir(lucidflux_dst.parent)
    ensure_dir(lucidnft_lora_dir)
    ensure_dir(proj_head_dst.parent)

    flow_src = hf_hub_download(FLUX_REPO, FLOW_FILE)
    ae_src = hf_hub_download(FLUX_REPO, AE_FILE)
    swinir_src = hf_hub_download(SWINIR_REPO, SWINIR_FILE)
    snapshot_download(ULTRAFLUX_REPO, allow_patterns="vae/*", local_dir=str(ultraflux_dir), local_dir_use_symlinks=False)
    snapshot_download(SIGLIP_REPO, local_dir=str(siglip_dir), local_dir_use_symlinks=False)
    snapshot_download(
        LUCIDNFT_REPO,
        allow_patterns=[f"{LUCIDNFT_TRANSFORMER_SUBDIR}/*", f"{LUCIDNFT_CONDITION_SUBDIR}/*"],
        local_dir=str(lucidnft_lora_dir),
        local_dir_use_symlinks=False,
    )
    snapshot_download(QWEN_EMBEDDING_REPO, local_dir=str(qwen_embedding_dir), local_dir_use_symlinks=False)
    lucidflux_src = hf_hub_download(LUCIDFLUX_REPO, LUCIDFLUX_FILE)
    prompt_embeddings_src = hf_hub_download(LUCIDFLUX_REPO, PROMPT_EMBEDDINGS_FILE)
    proj_head_src = hf_hub_download(LUCIDNFT_REPO, LUCIDCONSISTENCY_PROJ_HEAD_FILE)

    if args.force or not flow_dst.exists():
        flow_dst.write_bytes(Path(flow_src).read_bytes())
    if args.force or not ae_dst.exists():
        ae_dst.write_bytes(Path(ae_src).read_bytes())
    if args.force or not swinir_dst.exists():
        swinir_dst.write_bytes(Path(swinir_src).read_bytes())
    if args.force or not lucidflux_dst.exists():
        lucidflux_dst.write_bytes(Path(lucidflux_src).read_bytes())
    if args.force or not prompt_embeddings_dst.exists():
        prompt_embeddings_dst.write_bytes(Path(prompt_embeddings_src).read_bytes())
    if args.force or not proj_head_dst.exists():
        proj_head_dst.write_bytes(Path(proj_head_src).read_bytes())

    write_env(env_path, flow_dst, ae_dst)
    if args.print_env:
        l1, l2 = env_lines(flow_dst, ae_dst)
        sys.stdout.write("\n".join([l1, l2]) + "\n")

    write_manifest(
        manifest_path,
        {
            "model": MODEL_KEY,
            "flow_repo": FLUX_REPO,
            "flow_file": FLOW_FILE,
            "ae_repo": FLUX_REPO,
            "ae_file": AE_FILE,
            "swinir_repo": SWINIR_REPO,
            "swinir_file": SWINIR_FILE,
            "lucidflux_repo": LUCIDFLUX_REPO,
            "lucidflux_file": LUCIDFLUX_FILE,
            "prompt_embeddings_file": PROMPT_EMBEDDINGS_FILE,
            "lucidnft_repo": LUCIDNFT_REPO,
            "lucidnft_transformer_subdir": LUCIDNFT_TRANSFORMER_SUBDIR,
            "lucidnft_condition_subdir": LUCIDNFT_CONDITION_SUBDIR,
            "lucidconsistency_proj_head_file": LUCIDCONSISTENCY_PROJ_HEAD_FILE,
            "qwen_embedding_repo": QWEN_EMBEDDING_REPO,
            "qwen_embedding_dirname": QWEN_EMBEDDING_DIRNAME,
            "ultraflux_repo": ULTRAFLUX_REPO,
            "ultraflux_subdir": "vae",
            "siglip_repo": SIGLIP_REPO,
        },
    )

    sys.stdout.write("done.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
