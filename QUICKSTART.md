# 🚀 快速开始指南

**截止日期：11月9日 23:59**  
**剩余时间：9天**

---

## ⚡ 今晚必须完成（11月1日）

### 1. 环境安装（30分钟）

```bash
# 方案A：有NVIDIA GPU（推荐）
conda create -n hep2 python=3.9 -y
conda activate hep2
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install pandas matplotlib seaborn scikit-learn tqdm

# 方案B：没有GPU（用Colab）
# 打开 https://colab.research.google.com
# 运行时 → 更改运行时类型 → GPU
```

### 2. 准备数据（10分钟）

```bash
# 解压数据集
unzip "Three Datasets for Training, Validation and Test.zip"

# 整理文件结构
data/
├── train/       # 确保这里有图像
├── val/         # 确保这里有图像
├── test/        # 确保这里有图像
└── gt_training.csv
```

### 3. 测试代码（10分钟）

```bash
# 检查数据
python check_data.py

# 测试能否跑通（1个epoch）
# 修改 train.py 中的 num_epochs=1
python train.py
```

**如果上面都成功了，今晚任务完成！✅**

---

## 📅 每日计划

### Day 1-2（11月1-2日）- 基础训练
- [x] 环境搭建
- [ ] 运行baseline（无数据增强，10 epochs）
- [ ] 记录结果

### Day 3-5（11月3-5日）- 完整实验
- [ ] 有数据增强训练（30 epochs）
- [ ] 调整超参数（学习率、batch size）
- [ ] 对比实验结果

### Day 6-7（11月6-7日）- 结果分析
- [ ] 生成所有图表
- [ ] 分析混淆矩阵
- [ ] 准备Discussion内容

### Day 8（11月8日）- 写报告
- [ ] Introduction & Background
- [ ] Methodology
- [ ] Results
- [ ] Discussion & Conclusion

### Day 9（11月9日）- 最后检查
- [ ] 报告审阅
- [ ] 代码检查
- [ ] 提交！

---

## 🎯 最小可行方案（如果时间不够）

### 必做实验（2个）：
1. **Baseline**: 无数据增强，20 epochs
2. **Best**: 有数据增强，30 epochs

### 必需结果：
- 训练曲线（loss + accuracy）
- 混淆矩阵
- 每类准确率

### 报告长度：
- 10-12页足够
- 重点在Results（50分）

---

## 💻 运行命令速查

```bash
# 1. 检查数据
python check_data.py

# 2. 训练（默认配置）
python train.py

# 3. 测试
python test.py

# 4. 修改训练参数
# 编辑 train.py 中的 config 字典

# 5. 查看结果
# models/training_curves.png  - 训练曲线
# results/confusion_matrix.png - 混淆矩阵
```

---

## ⚙️ 常用配置修改

### 快速测试（1 epoch）
```python
# train.py
config = {
    ...
    'num_epochs': 1,
}
```

### 完整训练（30 epochs + 数据增强）
```python
# train.py
config = {
    ...
    'num_epochs': 30,
    'augment': True,
}
```

### Baseline（无数据增强）
```python
# train.py
config = {
    ...
    'num_epochs': 20,
    'augment': False,
}
```

### 减少batch size（如果内存不够）
```python
# train.py
config = {
    ...
    'batch_size': 16,  # 改为16或8
}
```

---

## 🔍 预期结果参考

### 训练时间：
- **有GPU**: ~30-60分钟（30 epochs）
- **Colab GPU**: ~30-60分钟
- **CPU**: ~3-6小时（不推荐）

### 准确率：
- **无数据增强**: ~89%
- **有数据增强**: ~96%

### 文件大小：
- 模型: ~230MB
- 训练曲线: ~100KB
- 混淆矩阵: ~200KB

---

## ❓ 常见错误解决

### 错误1: "CUDA out of memory"
```python
# 解决：减小batch_size
config = {
    'batch_size': 16,  # 从32改到16
}
```

### 错误2: "FileNotFoundError"
```bash
# 检查路径
python check_data.py

# 确保数据结构正确
data/train/00001.png  ✓
data/gt_training.csv  ✓
```

### 错误3: "Import Error"
```bash
# 重新安装依赖
pip install -r requirements.txt
```

---

## 📊 报告写作快速模板

### Introduction (1页)
> HEp-2细胞分类对自身免疫疾病诊断很重要。本项目使用AlexNet进行分类...

### Methodology (2页)
- AlexNet架构
- 数据预处理（resize to 256×256, crop to 224×224）
- 数据增强（rotation, flip, color jitter）
- 训练设置（SGD, lr=0.001, 30 epochs）

### Results (5页) ⭐ **重点**
- Table: 对比无/有数据增强的准确率
- Figure: 训练曲线（loss + acc）
- Figure: 混淆矩阵
- Table: 每个类别的precision/recall/F1
- 分析：哪些类别容易混淆？

### Discussion (2页)
- 结果分析
- 数据增强的作用
- 局限性和改进方向

---

## 🎉 Tips

1. **早点开始跑实验**：训练需要时间
2. **多保存checkpoint**：防止中断
3. **及时记录结果**：别忘了实验设置
4. **Results最重要**：占50分！
5. **图表要清晰**：用高分辨率

---

## 🆘 紧急联系

遇到问题？
1. 先查看 `README.md`
2. 运行 `check_data.py` 检查数据
3. 检查代码注释
4. Google搜索错误信息

---

**加油！你可以的！💪**
