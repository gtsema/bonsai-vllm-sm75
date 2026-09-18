# bonsai-vllm

Serves [Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit), Prism ML's ternary Qwen3.8-27B, on vLLM with custom CUDA kernels. Unofficial; not affiliated with Prism ML.

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 -v bonsai:/cache fraserpricee/bonsai-vllm:20260918
```

That pulls the image, downloads the 9.6 GB [weights](https://huggingface.co/fraserprice/Ternary-Bonsai-2-27B-vllm) into the `bonsai` volume, and serves an OpenAI-compatible API for `Bonsai-2-27B` on port 8000 with 262K context, tool calling, reasoning and MTP speculative decoding. Arguments after the image name go to `vllm serve` and override the defaults, e.g. `--max-model-len 32768`. The first start takes about 3 minutes (download aside) while vLLM compiles and captures CUDA graphs.

The image is built and tested for the RTX PRO 6000 Blackwell only; other NVIDIA GPUs are untested and may not work. If you hit a problem, please [open an issue](https://github.com/fraserprice/bonsai-vllm/issues).

## Throughput

Aggregate tokens/s across all concurrent requests on one RTX PRO 6000 Blackwell Max-Q (96 GB), ~2K generated tokens per request:

| Prompt | Concurrent requests | Prefill | Decode |
| --- | --- | --- | --- |
| 1K tokens | 1 | 2,150 | 198 |
| 1K tokens | 4 | 3,808 | 350 |
| 1K tokens | 8 | 3,990 | 515 |
| 10K tokens | 1 | 4,014 | 115 |
| 10K tokens | 4 | 4,089 | 222 |
| 10K tokens | 8 | 4,089 | 262 |

## Without Docker

Into an environment with vLLM 0.25.1 and the CUDA 13 toolkit (`nvcc`, which compiles the kernels on first use):

```bash
pip install git+https://github.com/fraserprice/bonsai-vllm
vllm serve fraserprice/Ternary-Bonsai-2-27B-vllm --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":7}'
```

## How it works

`prism_ternary` is a vLLM quantization plugin, under `src/prism_ternary`:

- `quant.py` registers the `prism_ternary` method: linear layers and the LM head hold 2-bit codes (16 per int32) with one FP16 scale per group of 128, re-laid out at load into the blocks the kernel reads. The MTP drafter runs in online FP8.
- `kernels.cu` has the two CUDA kernels: the blockwise signed Hadamard transform (block 1024) that rotates activations into the weights' basis, and a ternary GEMM that consumes the packed weights directly for up to 64 rows (decode and MTP verification).
- `kernels.py` wraps them as one `torch.compile`-friendly custom op; larger batches (prefill) dequantize with a Triton kernel and use a dense matmul.
- `convert.py` (`prism-ternary-convert --pack <mlx pack> --base <Qwen3.8-27B> --out <dir>`) builds the vLLM checkpoint from Prism ML's MLX pack, and checks every tensor's layout against the base model as it goes. Only the MTP head, which the pack doesn't carry, comes from the base model.

## Credits

- [Prism ML](https://prismml.com) for Bonsai 2 27B. Created using Bonsai by Prism ML.
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) for Qwen3.8-27B, which Bonsai is built from.
- [vLLM](https://github.com/vllm-project/vllm), which the image is built on.

Apache 2.0, as are all of the above. See `LICENSE` and `NOTICE`.
