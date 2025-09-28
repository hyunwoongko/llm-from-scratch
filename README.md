# rlhf-from-scratch
This project aims to perform RLHF training from scratch, implementing almost all components manually except for PyTorch and Ray. Each module is a minimal, educational reimplementation of large-scale systems focusing on clarity and core concepts rather than production readiness. It includes SFT (Supervised Fine Tuning), RLVR (Reinforcement Learning with Verifiable Reward), and an evaluation pipeline. A small version of Qwen3 model and opensource math dataset are used for training.

## Contents

| Category     | Description                                                | Reference                                                                        |
|--------------|------------------------------------------------------------|----------------------------------------------------------------------------------|
| datasets     | Minimal implementation of Hugging Face Datasets            | [datasets](https://github.com/huggingface/datasets)                              |
| rl           | Minimal implementation of PPO algorithm                    | [verl](https://github.com/volcengine/verl)                                       |
| inference    | Minimal implementation of an inference engine like vLLM    | [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)                         |
| parallelism  | Minimal implementation of model parallelism algorithms     | [oslo](https://github.com/EleutherAI/oslo)                                       |
| kernel       | Various kernel implementations including Flash Attention   | [trident](https://github.com/kakaobrain/trident)                                 |
| notebook     | Educational materials for each component written in Korean | [large-scale-lm-tutorials](https://github.com/tunib-ai/large-scale-lm-tutorials) |
