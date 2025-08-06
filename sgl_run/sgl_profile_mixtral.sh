# set trace path
export SGLANG_TORCH_PROFILER_DIR=/data/hbb2

# start server
#python -m sglang.launch_server --model-path /data/hbb2/model/Qwen1.5-MoE-A2.7B-Chat

# send profiling request from client
python -m sglang.bench_serving --backend sglang --model /data/hbb2/model/Mixtral-8x7B-Instruct-v0.1  --num-prompts 100 --sharegpt-output-len 100 --profile
