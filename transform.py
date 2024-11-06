import torch

# 加载 .pth 文件
model = torch.load('tracking/weights/resnet50_msmt17.pt')

print(model)

# 保存为 .pt 文件
# torch.save(model, 'tracking/weights/berry_reid_resnet50.pt')
