# set trace path
export SGLANG_TORCH_PROFILER_DIR=/data/hbb2/profile_log

# start server
#python -m sglang.launch_server --model-path /data/hbb2/model/Meta-Llama-3-8B-Instruct

# send profiling request from client
python -m sglang.bench_serving --backend sglang --model /data/hbb2/model/Meta-Llama-3-8B-Instruct --dataset-path /data/hbb2/datasets/wikitext_2_v1_test.json --num-prompts 100 --sharegpt-output-len 100 --profile
#--dataset-path /data/hbb2/datasets/wikitext_test.json