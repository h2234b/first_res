# set trace path
export SGLANG_TORCH_PROFILER_DIR=/data/hbb2/profile_log

# start server
#python -m sglang.launch_server --model-path /data/hbb2/model/Qwen1.5-MoE-A2.7B-Chat

# send profiling request from client
python -m sglang.bench_serving --backend sglang --model /data/hbb2/model/Qwen1.5-MoE-A2.7B-Chat --dataset-name random --num-prompts 100 --random-output-len 100 --profile
#--dataset-path /data/hbb2/datasets/gsm8k --num-prompts 100 --gen-output-len 100 --profile

#--dataset-path /data/hbb2/datasets/wikitext_test.json

#python your_script.py --dataset_name new_dataset --dataset_path /data/hbb2/datasets/new_dataset \
    #--gen_output_len 512 --gen_question_len 256
