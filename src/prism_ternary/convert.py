import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from safetensors import safe_open
from safetensors.torch import save_file

from prism_ternary.kernels import GROUP, HADAMARD_BLOCK, LANES
from prism_ternary.quant import QUANT_METHOD, pack_signs

AUX_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json")
VERIFY_MODULES = (
    "model.layers.0.linear_attn.in_proj_qkv",
    "model.layers.0.linear_attn.in_proj_z",
    "model.layers.0.linear_attn.out_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.v_proj",
    "model.layers.3.self_attn.o_proj",
    "lm_head",
)
VERIFY_ROWS = 4096
MIN_COSINE = 0.6
UNIT_OFFSET_NORMS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "language_model.norm.weight",
)
MIN_DENSE_COSINE = 0.9
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


class ConvertArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    pack: Path
    base: Path
    out: Path
    device: str


class PackedModule(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    words: torch.Tensor
    scales: torch.Tensor


def hf_name(path: str) -> str:
    return path if path.startswith("lm_head") else "model.language_model." + path.removeprefix("model.")


def hadamard(n: int, device: torch.device) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    two = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device)
    while h.shape[0] < n:
        h = torch.kron(h, two)
    return h / (n**0.5)


def dequantize(words: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows = words.shape[0]
    shifts = torch.arange(LANES, device=words.device, dtype=torch.int32) * 2
    codes = ((words[:, :, None] >> shifts) & 3).reshape(rows, -1).float() - 1.0
    return (codes.view(rows, -1, GROUP) * scales.float()[..., None]).reshape(rows, -1)


def rotate(weight: torch.Tensor, signs: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    rows = weight.shape[0]
    return ((weight.float() * signs).view(rows, -1, HADAMARD_BLOCK) @ h).reshape(rows, -1)


def verify(name: str, packed: PackedModule, base: torch.Tensor, signs: torch.Tensor, h: torch.Tensor) -> None:
    rows = min(VERIFY_ROWS, base.shape[0])
    device = h.device
    ternary = dequantize(packed.words[:rows].to(device), packed.scales[:rows].to(device))
    reference = rotate(base[:rows].to(device), signs, h)
    cosine = torch.nn.functional.cosine_similarity(ternary, reference, dim=1).mean().item()
    print(f"verify {name}: row cosine vs rotated base = {cosine:.4f}")
    if cosine < MIN_COSINE:
        raise ValueError(f"{name}: layout mismatch against base checkpoint (cosine {cosine:.4f})")


def dense(name: str, stored: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    shifted = stored.float() - 1.0 if name.endswith(UNIT_OFFSET_NORMS) else stored
    tensor = (shifted.movedim(2, 1) if name.endswith("conv1d.weight") else shifted).to(base.dtype).contiguous()
    if tensor.shape != base.shape:
        raise ValueError(f"{name}: shape {tuple(tensor.shape)} differs from base {tuple(base.shape)}")
    cosine = torch.nn.functional.cosine_similarity(tensor.float().flatten(), base.float().flatten(), dim=0).item()
    if cosine < MIN_DENSE_COSINE:
        raise ValueError(f"{name}: layout mismatch against base checkpoint (cosine {cosine:.4f})")
    return tensor


def load_packed(pack: safe_open, path: str, signs: dict[int, torch.Tensor]) -> PackedModule:
    words = torch.from_numpy(pack.get_tensor(f"{path}.weight").view(np.int32))
    scales = torch.from_numpy(pack.get_tensor(f"{path}.scales"))
    biases = torch.from_numpy(pack.get_tensor(f"{path}.biases"))
    stored_signs = torch.from_numpy(pack.get_tensor(f"{path}.signs"))
    width = words.shape[1] * LANES
    if not torch.equal(biases, -scales):
        raise ValueError(f"{path}: biases are not the negated scales")
    if not torch.equal(stored_signs, signs[width]):
        raise ValueError(f"{path}: sign vector differs from hadamard.json width {width}")
    return PackedModule(words=words, scales=scales)


def shard_of(name: str) -> int:
    match = LAYER_RE.match(name)
    return 2 if match is None else int(match.group(1)) // 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = ConvertArgs(**vars(parser.parse_args()))
    device = torch.device(args.device)

    pack_config = json.loads((args.pack / "config.json").read_text())
    hadamard_meta = json.loads((args.pack / "hadamard.json").read_text())
    if hadamard_meta["prism.hadamard.block_size"] != HADAMARD_BLOCK:
        raise ValueError("unexpected Hadamard block size")
    widths: list[int] = hadamard_meta["prism.hadamard.sign_widths"]
    values = np.asarray(hadamard_meta["prism.hadamard.sign_values"], dtype=np.float32)
    offsets = np.cumsum([0, *widths])
    signs = {
        width: torch.from_numpy(values[start:stop].copy())
        for width, start, stop in zip(widths, offsets[:-1], offsets[1:], strict=True)
    }
    base_index = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    modules = {m["path"]: bool(m["embedding"]) for m in pack_config["modules"]}
    h = hadamard(HADAMARD_BLOCK, device)

    def base_tensor(name: str) -> torch.Tensor:
        with safe_open(args.base / base_index[name], framework="pt") as f:
            return f.get_tensor(name)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(args.pack / "model.safetensors", framework="np") as pack:
        for path, is_embedding in modules.items():
            packed = load_packed(pack, f"language_model.{path}", signs)
            target = hf_name(path)
            if is_embedding:
                table = dequantize(packed.words.to(device), packed.scales.to(device))
                rows = table.shape[0]
                width = table.shape[1]
                table = (table.view(rows, -1, HADAMARD_BLOCK) @ h).reshape(rows, width) * signs[width].to(device)
                tensors[f"{target}.weight"] = table.to(torch.bfloat16).cpu()
                continue
            if path in VERIFY_MODULES:
                verify(path, packed, base_tensor(f"{target}.weight"), signs[packed.words.shape[1] * LANES].to(device), h)
            tensors[f"{target}.weight"] = packed.words
            tensors[f"{target}.scales"] = packed.scales
        quantized = {f"language_model.{path}.{part}" for path in modules for part in ("weight", "scales", "biases", "signs")}
        stored = {
            hf_name(key.removeprefix("language_model.")): key
            for key in pack.keys()
            if key.startswith("language_model.") and key not in quantized
        }
        tensors |= {name: dense(name, torch.from_numpy(pack.get_tensor(key)), base_tensor(name)) for name, key in stored.items()}
    print(f"kept {len(stored)} dense tensors from the pack")

    tensors |= {
        name: base_tensor(name)
        for name in base_index
        if not name.startswith("model.visual.") and name not in tensors
    }

    args.out.mkdir(parents=True, exist_ok=True)
    shard_names = {i: f"model-{i + 1:05d}-of-00003.safetensors" for i in range(3)}
    for shard, file in shard_names.items():
        save_file({k: v.contiguous() for k, v in tensors.items() if shard_of(k) == shard}, args.out / file, metadata={"format": "pt"})
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in tensors.values())},
        "weight_map": {k: shard_names[shard_of(k)] for k in tensors},
    }
    (args.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    config = json.loads((args.base / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": QUANT_METHOD,
        "bits": 2,
        "group_size": GROUP,
        "hadamard_block": HADAMARD_BLOCK,
        "signs": {str(width): pack_signs(vec.numpy()) for width, vec in signs.items()},
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    for aux in AUX_FILES:
        shutil.copyfile(args.base / aux, args.out / aux)
    print(f"wrote {len(tensors)} tensors to {args.out}")
