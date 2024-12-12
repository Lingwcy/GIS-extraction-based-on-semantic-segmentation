from PIL import Image
import torchvision.transforms as transforms
import torch
from nets.xception import Xception
from nets.setr.SETR import SETR_Naive_S
# 定义图像预处理步骤
transform = transforms.Compose([
    transforms.Resize((480, 480)),  # 调整图像大小以匹配网络输入
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 读取TIF图像
image_path = 'VOCdevkit/tif_dataset/src/3.tif'
image = Image.open(image_path).convert('RGB')  # 确保图像为RGB格式
# image.show()
# 应用预处理
image_tensor = transform(image)
print("单张图片输入shape:",image_tensor.shape)
# 添加批次维度
image_tensor = image_tensor.unsqueeze(0)
print("单张图片增加batch维度:",image_tensor.shape)
# 创建模型实例
aux_layers, model = SETR_Naive_S(dataset='my',_pe_type='learned')
model.eval()  # 设置为评估模式

# 前向传播
with torch.no_grad():
    output = model(image_tensor)

# # 处理输出
# print(output)


"""
输入参数： png,size(500x500),batch:1,num_classes:3
单张图片输入shape: torch.Size([3, 500, 500])
单张图片增加batch维度: torch.Size([1, 3, 500, 500])
输入形状 torch.Size([1, 3, 500, 500])
输入形状宽高HW 500 500
骨干网浅层形状: torch.Size([1, 24, 125, 125])
骨干网深层形状: torch.Size([1, 320, 32, 32])
aspp层输出形状: torch.Size([1, 256, 32, 32])
扩张骨干网浅层形状通道固定48: torch.Size([1, 48, 125, 125])
骨干网深层形状上采样形状: torch.Size([1, 256, 125, 125])
拼接形状 torch.Size([1, 256, 125, 125])
分类结果形状 torch.Size([1, 3, 125, 125])
最终上采样到H,W的结果 torch.Size([1, 3, 500, 500])
"""


"""
SETR_Naive_S

单张图片输入shape: torch.Size([3, 480, 480])
单张图片增加batch维度: torch.Size([1, 3, 480, 480])
输入特征x torch.Size([1, 3, 480, 480])
分割patches步骤1: torch.Size([1, 3, 30, 30, 16, 16])
分割patches步骤1: torch.Size([1, 3, 900, 256])
分割patches步骤2: torch.Size([1, 900, 256, 3])
分割patches步骤3: torch.Size([1, 900, 768])
分割patches步骤4: torch.Size([1, 900, 768])
decoder_reshape: torch.Size([1, 768, 30, 30])
conv1: torch.Size([1, 768, 30, 30])
bn1: torch.Size([1, 768, 30, 30])
act1: torch.Size([1, 768, 30, 30])
conv2: torch.Size([1, 10, 30, 30])
decoder_upsample: torch.Size([1, 10, 480, 480])

"""