# nanorlhf
This project aims to perform RLHF training from scratch, implementing almost all core components manually except for PyTorch. Each module is a minimal, educational reimplementation of large-scale systems focusing on clarity and core concepts rather than production readiness. This includes SFT and RL training pipeline with evaluation, for training a small Qwen3 model on an open-source math dataset.

## Contents

| Packages    | Description                                                | Reference                                                                                              |
|-------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `nanosets`  | Simplified implementation of pyarrow, HF datasets          | [pyarrow](https://github.com/apache/arrow), [datasets](https://github.com/huggingface/datasets)        |
| `nanoray`   | Simplified implementation of ray                           | [ray](https://github.com/ray-project/ray)                                                              |
| `nanovllm`  | Simplified implementation of vllm                          | [vllm](https://github.com/vllm-project/vllm), [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) |
| `nanoverl`  | Scratch implementation of SFT and PPO trainers             | [verl](https://github.com/volcengine/verl)                                                             |
| `nanotron`  | Scratch implementation of various parallelism algorithms   | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [oslo](https://github.com/EleutherAI/oslo)       |
| `kernels`   | Scratch implementation of various triton kernels           | [trident](https://github.com/kakaobrain/trident)                                                       |
| `notebooks` | Educational materials for each component written in Korean | [large-scale-lm-tutorials](https://github.com/tunib-ai/large-scale-lm-tutorials)                       |
