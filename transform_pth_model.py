import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict

# model_path = './tracking/weights/resnet50_berry_add_2.pth'
# model_path = 'resnet50_berry_add_3.pth'
model_path = 'resnet50_berry_add_6.pt'

state_dict = torch.load(model_path, map_location=torch.device('cuda'))
print(state_dict)
# 读取模型

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # if k.startswith("fc.add_block."):
    #     new_key = k.replace('fc.add_block.', 'fc.')
    if k.startswith("fc.fc.0."):
        new_key = k.replace('fc.fc.0.', 'classifier.')
    else:
        new_key = k
    new_state_dict[new_key] = v
print(new_state_dict)
# torch.save(new_state_dict, "resnet50_berry_add_6.pt")




