# from datasets import load_dataset
# from transformers import LlamaForCausalLM, LlamaTokenizer
# import torch
# from torch.profiler import profile, record_function, ProfilerActivity

# def run_inference():
#     # 加载 wikitext 数据集
#     dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

#     # 获取训练数据
#     train_data = dataset['train']

#     # 加载 Llama 模型和 tokenizer
#     model_name = "huggingface/llama-7b"  # 可以替换为其他模型
#     tokenizer = LlamaTokenizer.from_pretrained(model_name)
#     model = LlamaForCausalLM.from_pretrained(model_name)

#     # 将模型设置为评估模式
#     model.eval()

#     # 记录性能分析
#     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#         # 批量推理（以前五个样本为例）
#         for i in range(5):  # 你可以修改此处范围来调整输入样本数
#             input_text = train_data['context'][i]  # 假设 'context' 为数据集字段
#             inputs = tokenizer(input_text, return_tensors="pt")

#             # 进行推理
#             with torch.no_grad():
#                 output = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)

#             # 解码生成的文本
#             generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
#             # 打印原始输入和生成的文本
#             print(f"Input: {input_text}")
#             print(f"Generated: {generated_text}")

#     # 打印性能分析结果
#     print(prof.key_averages().table(sort_by="cpu_time_total"))

# # 运行推理和性能分析
# if __name__ == "__main__":
#     run_inference()

# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

MY_TOKEN="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"
# 加载 Llama 模型和 tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"  # 根据需要修改模型名称
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=MY_TOKEN)
model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=MY_TOKEN).to(device)

# 输入文本
input_text = "Once upon a time, in a faraway land,"

# 编码输入文本
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 生成输出
with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=50)

# 解码输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
