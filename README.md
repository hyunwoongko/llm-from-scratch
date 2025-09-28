# llm-from-scratch
This project aims to perform RL training by implementing most of the code from scratch, except for PyTorch and Ray. Many components are implemented as minimal versions that omit details and include only the core logic, and they are intended for educational purposes rather than use in real large-scale RL training.

## Contents
1. mini-datasets
   - simple implementation of huggingface datasets
   - reference: https://github.com/huggingface/datasets
2. mini-verl
   - simple implementation of verl
   - reference: https://github.com/volcengine/verl
3. mini-vllm
   - simple implementation of vllm
   - reference: https://github.com/GeeeekExplorer/nano-vllm
4. simple-megatron
   - simple implementation of tensor/pipeline parallelism and fsdp
   - reference: https://github.com/EleutherAI/oslo
4. custom triton kernels
   - various kernel implementation using triton
   - reference: https://github.com/kakaobrain/trident
5. jupyter notebooks
   - lecture materials written in Korean  
   - reference: https://github.com/tunib-ai/large-scale-lm-tutorials
