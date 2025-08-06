import json

dataset_path = "/data/hbb2/datasets/wikitext_2_v1_test.json"

with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 读取 JSON 文件

print(type(data))  # 应该是 list
print(len(data))   # 数据集大小
#print(data[0])     # 打印第一条数据
print(data.keys()) 

   
