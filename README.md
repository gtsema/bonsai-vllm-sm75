# Bonsai 2 27B via vLLM on RTX 2080 Ti (SM75)

## Что это

Форк `fraserprice/Ternary-Bonsai-2-27B-vllm`, адаптированный под инференс на NVIDIA **RTX 2080 Ti (22vram)** (compute capability 7.5 / SM75).

Оригинальный checkpoint vLLM рассчитан на новые GPU (SM80+): RTX 2080 Ti не поддерживает нативное BF16 и FlashAttention 2. В этой версии кастомное тринарное CUDA-расширение Prism было изменено, чтобы активации и MMA-пути работали в **FP16** на SM75. Веса по-прежнему тринарные. На практике модель работает через **vLLM 0.25.1** в Docker.

## Как запустить

Сборка образов:

```bash
sudo docker build --no-cache -t bonsai-vllm-sm75 .
```

Кэши (опционально, но настоятельно рекомендуется — без них каждый запуск скачивает модель и компилирует ядра заново):

```bash
sudo mkdir -p /opt/bonsai-cache/huggingface /opt/bonsai-cache/vllm
sudo chown -R $USER:$USER /opt/bonsai-cache
```

Запуск:

```bash
sudo docker run --gpus '"device=0"' -it --rm \
  --ipc=host \
  -p 8000:8000 \
  -v /opt/bonsai-cache/huggingface:/cache/huggingface \
  -v /opt/bonsai-cache/vllm:/cache/vllm \
  bonsai-vllm-sm75 \
  --model fraserprice/Ternary-Bonsai-2-27B-vllm \
  --served-model-name Bonsai-2-27B \
  --tensor-parallel-size 1 \
  --max-model-len 130000 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.95 \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

После запуска доступна OpenAI-совместимая API на `http://127.0.0.1:8000/v1/chat/completions`, с поддержкой tool calling. Первый старт медленный (компиляция едра, ~7 минут), дальше — быстрее благодаря кэшу.

## Сравнение с llama.cpp

| Runtime | Формат модели | Скорость | Контекст |
|---|---|---:|---:|
| vLLM (этот форк) | `Ternary-Bonsai-2-27B-vllm` | ~30 tok/s | ~130K |
| llama.cpp (PrismML) | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | ~35 tok/s | ~245K |

Оба варианта — это одна и та же модель (Bonsai 2 27B), только разные представления весов под разные рантаймы; слово в слово выводы между ними не совпадут.
---

# Bonsai 2 27B with vLLM on RTX 2080 Ti (SM75)

## What it is

A fork of `fraserprice/Ternary-Bonsai-2-27B-vllm` adapted for inference on **NVIDIA RTX 2080 Ti (22vram)** (compute capability 7.5 / SM75).

The upstream vLLM checkpoint targets newer GPUs (SM80+): the RTX 2080 Ti does not support native BF16 or FlashAttention 2. In this version the custom Prism ternary CUDA extension was changed so the activation and MMA paths run in **FP16** on SM75. The weights remain ternary. The model runs through **vLLM 0.25.1** in Docker.

## How to run

Build the image:

```bash
sudo docker build --no-cache -t bonsai-vllm-sm75 .
```

Caches (optional but strongly recommended — without them every run re-downloads the model and recompiles kernels):

```bash
sudo mkdir -p /opt/bonsai-cache/huggingface /opt/bonsai-cache/vllm
sudo chown -R $USER:$USER /opt/bonsai-cache
```

Launch:

```bash
sudo docker run --gpus '"device=0"' -it --rm \
  --ipc=host \
  -p 8000:8000 \
  -v /opt/bonsai-cache/huggingface:/cache/huggingface \
  -v /opt/bonsai-cache/vllm:/cache/vllm \
  bonsai-vllm-sm75 \
  --model fraserprice/Ternary-Bonsai-2-27B-vllm \
  --served-model-name Bonsai-2-27B \
  --tensor-parallel-size 1 \
  --max-model-len 130000 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.95 \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

An OpenAI-compatible API is exposed on `http://127.0.0.1:8000/v1/chat/completions`, with tool calling enabled. The first startup is slow (kernel compilation, ~7 minutes); subsequent ones are faster thanks to the cache.

## Comparison with llama.cpp

| Runtime | Model format | Speed | Context |
|---|---|---:|---:|
| vLLM (this fork) | `Ternary-Bonsai-2-27B-vllm` | ~30 tok/s | ~130K |
| llama.cpp (PrismML) | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | ~35 tok/s | ~245K |

