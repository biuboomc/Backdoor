import torch

# 假设模型文件路径
model_path = './anp/bd.pth'

# 加载模型状态字典
model_state_dict = torch.load(model_path)

for key, value in model_state_dict.items():
    print(f"Key: {key}, Value: {value}")

