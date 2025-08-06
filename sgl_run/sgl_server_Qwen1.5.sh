# set trace path
export SGLANG_TORCH_PROFILER_DIR=/data/hbb2/profile_log

# start server
python -m sglang.launch_server --model-path /data/hbb2/model/Qwen1.5-MoE-A2.7B-Chat

# send profiling request from client
#python -m sglang.bench_serving --backend sglang --model-path meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile#