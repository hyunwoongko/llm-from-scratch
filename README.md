# rlhf-from-scratch
This project aims to perform RLHF training from scratch, implementing almost all components manually except for PyTorch and Ray. Each module is a minimal, educational reimplementation of large-scale systems focusing on clarity and core concepts rather than production readiness. It includes SFT (Supervised Fine Tuning), RLVR (Reinforcement Learning with Verifiable Reward), and an evaluation pipeline. A small version of Qwen3 model and opensource math dataset are used for training.

## Contents
1. datasets
   - minimal implementation of huggingface datasets
   - reference: https://github.com/huggingface/datasets
2. rl
   - minimal implementation of ppo algorithm
   - reference: https://github.com/volcengine/verl
3. inference
   - minimal implementation of inference engine similar with vllm
   - reference: https://github.com/GeeeekExplorer/nano-vllm
4. parallelism
   - minimal implementation of model parallelism algorithms
   - reference: https://github.com/EleutherAI/oslo
5. kernels
   - various kernel implementation including flash attention using triton
   - reference: https://github.com/kakaobrain/trident
6. notebooks
   - educational materials of each component written in Korean  
   - reference: https://github.com/tunib-ai/large-scale-lm-tutorials
