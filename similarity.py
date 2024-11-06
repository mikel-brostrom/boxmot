import torch
from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# 配置路径
# model_path = './tracking/weights/osnet_x0_25_msmt17.pt'
model_path = './tracking/weights/resnet50_berry_add.pth'

img1_path = './img/0015/0015_L2_2_Ripe7_39_536frame.jpg'
img2_path = './img/0015/0015_L2_2_Ripe7_39_540frame.jpg'

# 1. 加载模型架构
model = build_model(name='osnet_x0_25', num_classes=751, loss='softmax')  # num_classes

# 2. 加载预训练权重
load_pretrained_weights(model, model_path)

# 3. 设置为评估模式
model.eval()

# 4. 将模型移到 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize((256, 128)),  # 行人重识别常用输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])


# 6. 图像加载和预处理
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)  # 增加批次维度并移动到设备


image1 = preprocess_image(img1_path)
image2 = preprocess_image(img2_path)


# 7. 特征提取函数
def extract_features(image_tensor):
    with torch.no_grad():  # 不需要梯度计算
        features = model(image_tensor)
    return features


# 8. 提取图像特征
features1 = extract_features(image1)
features2 = extract_features(image2)


# 9. 计算余弦相似度
def cosine_similarity(feature1, feature2):
    feature1 = F.normalize(feature1, p=2, dim=1)  # 特征归一化
    feature2 = F.normalize(feature2, p=2, dim=1)
    similarity = torch.mm(feature1, feature2.t())  # 计算余弦相似度
    return similarity.item()


# 10. 计算并输出相似度
similarity = cosine_similarity(features1, features2)
print(f"图像相似度: {similarity:.4f}")
