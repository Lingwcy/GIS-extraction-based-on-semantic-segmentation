# 耕地地块分割任务中模型参数量与结构的比较分析


本项目针对耕地地块分割任务，比较分析了不同深度学习模型的参数量和结构对分割效果的影响。主要包括基于CNN、CNN+注意力机制以及纯自注意力机制的模型。

## 项目概述

### 研究目的
1. 研究模型参数量对于简单分割任务结果的影响
2. 研究卷积结构和自注意力结构分别对于耕地地块提取任务结果的影响

### 数据集
使用HRSCD高分辨率语义变化检测遥感数据集：
- 来源：法国国家地理和森林信息研究所(IGN)的BD ORTHO数据库
- 分辨率：每像素50厘米
- 波段：RGB三个波段
- 地貌类型：包含裸地、人造区、农业用地、森林、湿地和水域等六种类型
- 标注：包含使用中的耕地(1)和暂未使用的耕地(2)两类标签

### 模型架构
本项目实现和比较了以下几种模型：

1. 基于CNN的模型:
- DeepLabV3+ MobileNetV2
- DeepLabV3+ Xception
- DeepLabV3+ ResNet101
- DeepLabV3+ SE_ResNet101
- MiniNetV3 (本项目提出的轻量化模型)

2. 基于Transformer的模型:
- SETR_Naive_S


## 环境要求 
python
python 3.8
cuda 11.8
torch 2.4.1
RTX4060 8G

## 项目结构
├── logs/ # 日志（训练和预测的日志）
├── nets/ # 网络模型定义
├── utils/ # 工具函数
├── test_data/ # 测试数据
├── segment_anything # sam2库
├── VOCdevkit # 数据集
├── train.py # 训练脚本
├── setr_train.py # SETR模型训练脚本
├── MiniNet_train.py # MiniNet模型训练脚本
├── deeplab.py # deeplabv3+模型配置脚本
├── MiniNet.py # MiniNet模型配置脚本
├── setr.py # setr模型配置脚本
├── predict.py # 预测脚本
└── parameters_count.py # 参数统计脚本

## 实验结果

各模型性能对比:

| 模型 | 参数量(百万) | mIOU | mPA | mPrecision | mRecall |
|------|------------|------|-----|------------|---------|
| DeepLabV3+ MobileNetV2 | 5.82M | 87.26% | 93.01% | 93.32% | 93.01% |
| DeepLabV3+ Xception | 54.71M | 81.16% | 89.54% | 89.57% | 89.54% |
| DeepLabV3+ ResNet101 | 59.34M | 83.18% | 90.63% | 90.93% | 90.63% |
| DeepLabV3+ SE_ResNet101 | 64.09M | 83.27% | 90.69% | 90.97% | 90.69% |
| SETR_Naive_S | 88.00M | 75.25% | 85.34% | 86.47% | 85.34% |
| MiniNetV3 | 0.13M | 82.64% | 90.29% | 90.62% | 90.29% |


## 主要结论

1. 对于简单分割任务，低参数量的模型可能因为其较低的复杂度而表现更好，用复杂的模型去模拟简单的分割任务可能会加剧过拟合风险。

2. SE(Squeeze-and-Excitation)模块的效果：在这组测试中，SE_Resnet101的参数量比Resnet101多，但是其在所有指标上的表现都比Resnet101好。这表明SE模块在提升模型的分割效果上是有效的，即使在简单任务上。

3. Transformer结构的模型在简单任务上可能不如基于卷积神经网络的模型有效。


## 权重和日志
通过网盘分享的文件：logs.rar等4个文件
链接: https://pan.baidu.com/s/1cH9b4aID8piaIVnkzZBcDw?pwd=9iws 提取码: 9iws

HRSCD.rar – 数据集源文件
Model_data .rar– 经过预训练的权重文件(放入项目model_data文件夹)
Logs.rar – 经过模型训练的日志及权重文件(放入项目logs文件夹)

环境需求在位于项目中的requirements.txt。项目结构位于项目中的ReadMe.md
## 引用项目
https://rcdaudt.github.io/hrscd/
https://github.com/bubbliiiing/pspnet-pytorch/tree/bilibili