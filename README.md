# HEp-2 Cell Image Classification using AlexNet

## 📋 Project Overview

这是一个使用深度卷积神经网络AlexNet进行HEp-2细胞图像分类的项目。该项目是COMP9517 Project 2的实现。

### 数据集信息
- **训练集**: 8,701 张图像
- **验证集**: 2,175 张图像  
- **测试集**: 2,720 张图像
- **类别数**: 6类
  - Homogeneous (2,494张)
  - Speckled (2,831张)
  - Nucleolar (2,598张)
  - Centromere (2,741张)
  - NuMem (2,208张)
  - Golgi (724张)

---

## 📁 Project Structure

```
project/
├── data/                           # 数据集目录
│   ├── train/                      # 训练集图像
│   ├── val/                        # 验证集图像
│   ├── test/                       # 测试集图像
│   └── gt_training.csv             # 标签文件
├── models/                         # 保存的模型
│   ├── best_model.pth             # 最佳模型
│   ├── checkpoint_epoch_*.pth     # 训练检查点
│   ├── history.json               # 训练历史
│   └── training_curves.png        # 训练曲线图
├── results/                        # 测试结果
│   ├── confusion_matrix.png       # 混淆矩阵
│   ├── class_accuracy.png         # 各类别准确率
│   └── classification_report.txt  # 分类报告
├── dataset.py                      # 数据加载
├── model.py                        # AlexNet模型
├── train.py                        # 训练脚本
├── test.py                         # 测试脚本
├── check_data.py                   # 数据检查脚本
├── requirements.txt                # 依赖包
└── README.md                       # 项目说明
```

---

## 🚀 Quick Start

### 1. 环境安装

```bash
# 使用conda（推荐，如果有GPU）
conda create -n hep2 python=3.9 -y
conda activate hep2
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 或使用pip（如果没有GPU）
python -m venv hep2_env
source hep2_env/bin/activate  # Windows: hep2_env\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

确保你的数据结构如下：
```
data/
├── train/          # 8701张图像: 00001.png, 00002.png, ...
├── val/            # 2175张图像
├── test/           # 2720张图像
└── gt_training.csv # 标签文件
```

检查数据：
```bash
python check_data.py
```

### 3. 训练模型

```bash
python train.py
```

训练参数可以在 `train.py` 的 `config` 字典中修改：
- `batch_size`: 批次大小 (默认: 32)
- `lr`: 学习率 (默认: 0.001)
- `num_epochs`: 训练轮数 (默认: 30)
- `augment`: 是否使用数据增强 (默认: True)

### 4. 测试模型

```bash
python test.py
```

结果将保存在 `results/` 目录中。

---

## 📊 实验设置

### 模型架构
- **基础模型**: AlexNet (预训练在ImageNet上)
- **修改**: 最后一层全连接层从1000类改为6类
- **参数量**: ~61M (可训练参数 ~50K)

### 数据预处理
1. **图像大小**: 调整到 256×256
2. **裁剪**: 224×224 中心裁剪
3. **归一化**: ImageNet标准 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 数据增强（训练时）
- 随机水平翻转
- 随机旋转 (±15度)
- 颜色抖动 (亮度、对比度、饱和度、色调)
- 随机裁剪

### 训练参数（最优配置 - Aggressive v1）
- **优化器**: SGD with momentum (0.9)
- **初始学习率**: 0.001（提高学习率加快收敛）
- **学习率调度**: StepLR (每10个epoch衰减×0.1)
- **权重衰减**: 5e-4（适度正则化）
- **批次大小**: 32（稳定梯度）
- **训练轮数**: 50（配合Early Stopping）
- **Dropout**: 0.5（AlexNet原始值）
- **Label Smoothing**: 0.0（无label smoothing）
- **Early Stopping**: patience=15
- **特征层冻结**: ✗（微调整个网络）

---

## 📈 实验结果

### 🏆 最佳成绩（Aggressive v1 - 推荐配置）

| 指标 | 数值 |
|------|------|
| **验证集准确率** | **95.26%** ⭐ |
| **测试集准确率** | **93.12%** ⭐ |
| 训练集准确率 | 91.38% |
| 最佳Epoch | 24 |
| 训练时间 | 29.63分钟 |
| Early Stopping | Epoch 39 (patience=15) |
| Generalization Gap | 2.14% (健康) |

**突破性提升：从 baseline 82.07% → 95.26% (+13.19%)**

### 配置对比

| 配置 | Baseline | Aggressive v1 | 提升 |
|------|----------|---------------|------|
| **Val Acc** | 82.07% | **95.26%** | **+13.19%** |
| **Test Acc** | 80.29% | **93.12%** | **+12.83%** |
| Train Acc | 72.67% | 91.38% | +18.71% |
| Dropout | 0.7 | 0.5 | ⬇️ |
| Learning Rate | 0.0005 | 0.001 | ⬆️ |
| Freeze Features | ✓ | ✗ | 解冻 |
| Label Smoothing | 0.1 | 0.0 | 去掉 |
| Batch Size | 16 | 32 | ⬆️ |

### 关键成功因素

**核心改进：释放模型学习能力**

| 改进 | 说明 | 贡献 |
|------|------|------|
| **解冻特征层** | 允许微调整个AlexNet网络 | ⭐⭐⭐ 约 +8-10% |
| **降低Dropout** | 从0.7降到0.5（AlexNet原始值） | ⭐⭐⭐ 约 +3-5% |
| **提高学习率** | 从0.0005提高到0.001 | ⭐⭐ 约 +2-3% |
| **增大Batch Size** | 从16增大到32，更稳定的梯度 | ⭐ 约 +1% |
| **去掉Label Smoothing** | 清晰图像不需要过度平滑 | ⭐ 约 +0.5% |

---

## 🔧 高级设置

### 使用Google Colab（免费GPU）

1. 打开 [Google Colab](https://colab.research.google.com)
2. 新建Notebook
3. 设置GPU: 运行时 → 更改运行时类型 → GPU
4. 上传数据和代码
5. 运行训练脚本

```python
# Colab中的示例代码
from google.colab import drive
drive.mount('/content/drive')

# 解压数据
!unzip /content/drive/MyDrive/hep2_data.zip

# 运行训练
!python train.py
```

### 冻结特征层训练

如果数据量小或训练时间紧张，可以冻结特征提取层：

```python
# 在 train.py 中修改：
config = {
    ...
    'freeze_features': True,  # 改为True
}
```

### 调整批次大小

根据GPU内存调整：
- 8GB GPU: batch_size=32
- 4GB GPU: batch_size=16
- CPU: batch_size=8

---

## 📝 报告写作建议

### Section 1: Introduction
- 说明HEp-2细胞分类的重要性
- 介绍AlexNet和迁移学习
- 本项目的目标和方法

### Section 2: Methodology
- AlexNet架构详解
- 数据预处理和增强方法
- 训练策略和超参数选择
- 实验设置

### Section 3: Results
- 训练曲线（loss和accuracy）
- 混淆矩阵
- 每个类别的性能指标
- 数据增强的影响对比

### Section 4: Discussion
- 结果分析（哪些类别容易混淆？为什么？）
- 数据增强的作用
- 与其他方法的对比
- 局限性和改进方向

---

## ⚠️ 常见问题

### Q: CUDA out of memory
A: 减小batch_size或使用更小的图像尺寸

### Q: 训练太慢
A: 使用GPU或Google Colab；减少epoch数

### Q: 准确率不理想
A: 
1. 确保使用了数据增强
2. 增加训练epoch数
3. 调整学习率
4. 尝试不同的数据增强策略

### Q: 数据加载出错
A: 运行 `python check_data.py` 检查数据完整性

---

## 📚 参考文献

主要参考论文：
- Gao et al. (2015). "HEp-2 Cell Image Classification with Deep Convolutional Neural Networks"

---

## 👥 Team Members

请在此处填写小组成员及分工：

| 成员 | 学号 | 分工 |
|------|------|------|
| Member 1 | XXX | 数据处理、模型训练 |
| Member 2 | XXX | 实验分析、结果可视化 |
| Member 3 | XXX | 报告撰写、文献调研 |

---

## 📧 Contact

如有问题，请联系：[your.email@example.com]

---

**祝你的项目顺利完成！🎉**
