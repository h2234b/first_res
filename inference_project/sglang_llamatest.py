


# # llama_inference.py
# from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

# @function
# def llama_inference(s, user_input):
#     s += system("You are a helpful assistant based on the Llama model.")
#     s += user(user_input)
#     s += assistant(gen("response", max_tokens=256))
#     return s["response"]

# # 设置SGLang服务器的地址和端口
# set_default_backend(RuntimeEndpoint("http://localhost:30000"))

# # 运行推理函数，并获取回答
# response = llama_inference.run(user_input="What is the capital of France?")
# print("Assistant:", response)
# # curl -X pst "hh" -H


from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
import torch
from torch.profiler import profile, record_function, ProfilerActivity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

@function
def llama_inference(s, user_input):
    s += system("You are a helpful assistant based on the Llama model.")
    s += user(user_input)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        record_shapes=True, 
        profile_memory=True,
        with_stack=True  # 记录函数调用栈
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_output')
    ) as prof:
        with record_function("model_inference"):
            # 假设这里的gen函数是SGLang中用于生成回答的函数
            # 并且它内部使用了PyTorch进行计算
            s += assistant(gen("response", max_tokens=256,))
    
    # 打印PyTorch profiler的结果
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # 将结果导出为Chrome Trace格式
    prof.export_chrome_trace('profile2.json')

set_default_backend(RuntimeEndpoint("http://localhost:30000"))
response = llama_inference.run(user_input="What is the capital of France?")
print("Assistant:", response)


