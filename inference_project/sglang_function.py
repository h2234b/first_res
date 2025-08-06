
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# 设置 sgLang 后端
set_default_backend(RuntimeEndpoint("http://localhost:30000"))

# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

MY_TOKEN="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"

# # 加载 Llama 模型和 tokenizer
# tokenizer = LlamaTokenizer.from_pretrained("meta/llama-7b")
# model = LlamaForCausalLM.from_pretrained("meta/llama-7b")

# 加载 Llama 模型和 tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"  
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=MY_TOKEN)
model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=MY_TOKEN).to(device)

# 定义推理函数
@function
def llama_inference(s, user_input):
    s += system("You are a helpful assistant based on the Llama model.")
    s += user(user_input)

    # 进行推理
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=256)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    s += assistant(generated_text)

    return s["response"]

# 执行推理
user_input = "What is the capital of France?"
response = llama_inference.run(user_input=user_input)
print("Assistant:", response)
