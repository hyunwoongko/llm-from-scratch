# RLHF from scratch
This project aims to perform RLHF training from scratch, implementing almost all components manually except for PyTorch and Ray. Each module is a minimal, educational reimplementation of large-scale systems focusing on clarity and core concepts rather than production readiness. This includes SFT and RL training pipeline with evaluation, for training a small Qwen3 model on an open-source math dataset.

## Contents

| Category     | Description                                                | Reference                                                                        |
|--------------|------------------------------------------------------------|----------------------------------------------------------------------------------|
| training     | Scratch implementation of SFT and PPO trainers             | [verl](https://github.com/volcengine/verl)                                       |
| parallelism  | Scratch implementation of various parallelism algorithms   | [oslo](https://github.com/EleutherAI/oslo)                                       |
| kernels      | Scratch implementation of various triton kernels           | [trident](https://github.com/kakaobrain/trident)                                 |
| datasets     | Simplified scratch implementation of Hugging Face Datasets | [datasets](https://github.com/huggingface/datasets)                              |
| inference    | Simplified scratch implementation of vLLM                  | [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)                         |
| notebooks    | Educational materials for each component written in Korean | [large-scale-lm-tutorials](https://github.com/tunib-ai/large-scale-lm-tutorials) |
