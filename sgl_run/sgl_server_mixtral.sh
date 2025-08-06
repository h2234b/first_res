# set trace path
export SGLANG_TORCH_PROFILER_DIR=/data/hbb2

# start server
python -m sglang.launch_server --model-path /data/hbb2/model/Mixtral-8x7B-Instruct-v0.1 --tp 2 --dp 2

# send profiling request from client
#python -m sglang.bench_serving --backend sglang --model-path meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile#