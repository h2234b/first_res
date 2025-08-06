import transformers
import torch
from modelscope import snapshot_download
from modelscope.msdatasets import MsDataset


model_id = "/data/hbb2/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

ds =  MsDataset.load('modelscope/wikitext', subset_name='wikitext-103-v1', split='test')
# 提取数据集中的示例
# 假设每一条样本是字符串类型，如果不是请调整
sample_texts = [item['text'] for item in ds[:5]]  # 提取前 5 条样本

# 构造对话消息
messages = []
for i, text in enumerate(sample_texts):
    messages.append({"role": "user", "content": text})


prompt = pipeline.tokenizer.apply_chat_template(
		messages, 
		tokenize=False, 
		add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])

# ##mistral
# import torch
# import time
# from modelscope import AutoModelForCausalLM,AutoTokenizer

# device="cuda"
# model=AutoModelForCausalLM.from_pretrained("/data/hbb2/Mistral-7B-Instruct-v0.2",torch_dtype=torch.float16)
# tokenizer=AutoTokenizer.from_pretrained("/data/hbb2/Mistral-7B-Instruct-v0.2")


# # # 包装模型以支持数据并行
# # model = torch.nn.DataParallel(model)
# # model.to(device)

# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

# model_inputs = encodeds.to(device)
# model.to(device)
# start_time = time.time()
# generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
# end_time = time.time()
# decoded = tokenizer.batch_decode(generated_ids)
# print(decoded[0])
# print(f"Time taken for generation: {end_time - start_time} seconds")
