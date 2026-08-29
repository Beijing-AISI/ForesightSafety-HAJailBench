#!/usr/bin/env bash

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$HOME/huggingface_cache"
export VLLM_USE_MULTIPROCESSING_SPAWN=1
export multiprocessing_start_method=spawn
export VLLM_DISTRIBUTED_EXECUTION_BACKEND=ray
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
vllm serve Qwen/Qwen3-14B -tp 1 --trust-remote-code --dtype auto --port 3114 --gpu-memory-utilization 0.6 --max-model-len 8192 --swap-space 16 --response-role "assistant" --chat-template "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}"
