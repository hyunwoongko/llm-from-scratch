# rlhf-from-scratch
This project aims to perform RLHF training from scratch, implementing almost all components manually except for PyTorch and Ray. Each module is a minimal, educational reimplementation of large-scale systems focusing on clarity and core concepts rather than production readiness. It includes SFT (Supervised Fine Tuning), RLVR (Reinforcement Learning with Verifiable Reward), and an evaluation pipeline. A small version of Qwen3 model and opensource math dataset are used for training.

## Contents
1. mini-datasets
   - a minimal implementation of huggingface datasets
   - reference: https://github.com/huggingface/datasets
2. mini-verl
   - a minimal implementation of verl
   - reference: https://github.com/volcengine/verl
3. mini-vllm
   - a minimal implementation of vllm
   - reference: https://github.com/GeeeekExplorer/nano-vllm
4. mini-megatron
   - a minimal implementation of tensor/pipeline parallelism and fsdp
   - reference: https://github.com/EleutherAI/oslo
5. custom triton kernels
   - various kernel implementation including flash attention using triton
   - reference: https://github.com/kakaobrain/trident
6. jupyter notebooks
   - educational materials of each component written in Korean  
   - reference: https://github.com/tunib-ai/large-scale-lm-tutorials
