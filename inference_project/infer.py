import torch
from torch.profiler import ProfilerActivity, profile
from torch.autograd.profiler import profile, ProfilerActivity
import requests

# 模型推理函数
def infer(model_url, input_text):
    headers = {"Content-Type": "application/json"}
    data = {
        "text": input_text,
        "sampling_params": {
            "max_new_tokens": 16,
            "temperature": 0
        }
    }
    response = requests.post(model_url, json=data, headers=headers)
    return response.json()

# 主函数
if __name__ == '__main__':
    model_url = "http://localhost:30000/generate"
    input_text = "Once upon a time,"

    # 使用PyTorch profiler监控性能
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        result = infer(model_url, input_text)

    # 打印分析结果
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # 将结果导出为Chrome Trace格式
    prof.export_chrome_trace('profile.json')