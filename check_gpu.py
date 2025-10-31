#!/usr/bin/env python3
"""检查GPU和PyTorch配置"""

import sys

print("="*60)
print("GPU 和 PyTorch 配置检查")
print("="*60 + "\n")

# 1. 检查PyTorch
print("1. 检查PyTorch安装...")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
except ImportError:
    print("  ✗ PyTorch 未安装！")
    sys.exit(1)

# 2. 检查CUDA
print("\n2. 检查CUDA支持...")
print(f"  PyTorch built with CUDA: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  ✓ GPU可用！")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  Current GPU: {torch.cuda.current_device()}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    
    # 测试GPU
    print("\n3. 测试GPU...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = x @ y
        print(f"  ✓ GPU计算测试成功！")
    except Exception as e:
        print(f"  ✗ GPU计算测试失败: {e}")
else:
    print(f"  ✗ GPU不可用！")
    print("\n  可能的原因：")
    print("    1. 没有NVIDIA显卡")
    print("    2. 没有安装NVIDIA驱动")
    print("    3. 安装了CPU版本的PyTorch")
    print("\n  解决方案：")
    print("    重新安装GPU版本的PyTorch（见下方）")

# 4. 推荐安装命令
print("\n" + "="*60)
print("推荐的PyTorch安装命令：")
print("="*60)
print("\n如果CUDA 11.8:")
print("  pip uninstall torch torchvision")
print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("\n如果CUDA 12.1:")
print("  pip uninstall torch torchvision")
print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print("\n或使用conda:")
print("  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")

print("\n如果不确定CUDA版本，运行: nvidia-smi")
print("="*60)
