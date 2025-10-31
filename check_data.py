import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def check_data_structure(data_root, csv_file):
    """检查数据集结构和完整性"""
    
    print("="*60)
    print("数据集检查报告")
    print("="*60 + "\n")
    
    # 1. 检查文件夹是否存在
    print("1. 检查文件夹结构...")
    folders = ['train', 'val', 'test']
    for folder in folders:
        folder_path = os.path.join(data_root, folder)
        if os.path.exists(folder_path):
            print(f"  ✓ {folder}/ 文件夹存在")
        else:
            print(f"  ✗ {folder}/ 文件夹不存在！")
            return False
    
    # 2. 检查CSV文件
    print("\n2. 检查CSV文件...")
    if not os.path.exists(csv_file):
        print(f"  ✗ CSV文件不存在: {csv_file}")
        return False
    print(f"  ✓ CSV文件存在: {csv_file}")
    
    # 3. 读取CSV并检查格式
    print("\n3. 分析CSV内容...")
    try:
        df = pd.read_csv(csv_file)
        print(f"  ✓ CSV读取成功，共 {len(df)} 条记录")
        print(f"  列名: {list(df.columns)}")
        
        # 检查类别
        if 'Image class' in df.columns:
            classes = df['Image class'].value_counts()
            print("\n  类别分布:")
            for cls, count in classes.items():
                print(f"    - {cls}: {count} 张")
        else:
            print("  ✗ 未找到 'Image class' 列")
    except Exception as e:
        print(f"  ✗ CSV读取失败: {e}")
        return False
    
    # 4. 检查图像文件
    print("\n4. 检查图像文件...")
    class_to_idx = {
        'Homogeneous': 0,
        'Speckled': 1,
        'Nucleolar': 2,
        'Centromere': 3,
        'NuMem': 4,
        'Golgi': 5
    }
    
    for folder in folders:
        folder_path = os.path.join(data_root, folder)
        
        # 统计该文件夹的图像
        img_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        print(f"\n  {folder}/ 文件夹:")
        print(f"    - 图像数量: {len(img_files)}")
        
        # 检查几个图像文件
        sample_imgs = img_files[:3] if len(img_files) >= 3 else img_files
        all_valid = True
        for img_file in sample_imgs:
            img_path = os.path.join(folder_path, img_file)
            try:
                img = Image.open(img_path)
                img_array = np.array(img)
                # print(f"    ✓ {img_file}: {img.size}, {img.mode}, shape={img_array.shape}")
            except Exception as e:
                print(f"    ✗ {img_file}: 无法读取 ({e})")
                all_valid = False
        
        if all_valid:
            print(f"    ✓ 样本图像检查通过")
        
        # 检查CSV中对应的标签
        matched = 0
        for img_file in img_files:
            img_id = int(img_file.replace('.png', ''))
            if img_id in df['Image ID'].values:
                matched += 1
        
        print(f"    - CSV中有标签的图像: {matched}/{len(img_files)}")
        if matched == len(img_files):
            print(f"    ✓ 所有图像都有对应标签")
        else:
            print(f"    ⚠ 有 {len(img_files) - matched} 张图像没有标签")
    
    # 5. 可视化一些样本
    print("\n5. 生成样本可视化...")
    visualize_samples(data_root, csv_file, class_to_idx)
    
    print("\n" + "="*60)
    print("✓ 数据检查完成！")
    print("="*60 + "\n")
    
    return True


def visualize_samples(data_root, csv_file, class_to_idx):
    """可视化每个类别的样本图像"""
    df = pd.read_csv(csv_file)
    
    # 为每个类别找一张训练集图像
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    train_path = os.path.join(data_root, 'train')
    train_imgs = set(os.listdir(train_path))
    
    for class_idx in range(6):
        class_name = idx_to_class[class_idx]
        
        # 找一张该类别的图像
        class_df = df[df['Image class'] == class_name]
        
        for _, row in class_df.iterrows():
            img_id = row['Image ID']
            img_filename = f"{img_id:05d}.png"
            
            if img_filename in train_imgs:
                img_path = os.path.join(train_path, img_filename)
                img = Image.open(img_path)
                
                axes[class_idx].imshow(img, cmap='gray' if img.mode == 'L' else None)
                axes[class_idx].set_title(f'{class_name}\n(ID: {img_id})', 
                                         fontsize=12, fontweight='bold')
                axes[class_idx].axis('off')
                break
    
    plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 样本图像已保存到: data_samples.png")
    plt.close()


def check_image_properties(data_root):
    """检查图像属性（大小、通道等）"""
    print("\n6. 检查图像属性...")
    
    train_path = os.path.join(data_root, 'train')
    img_files = [f for f in os.listdir(train_path) if f.endswith('.png')][:100]
    
    sizes = []
    modes = []
    
    for img_file in img_files:
        img_path = os.path.join(train_path, img_file)
        img = Image.open(img_path)
        sizes.append(img.size)
        modes.append(img.mode)
    
    # 统计
    unique_sizes = list(set(sizes))
    unique_modes = list(set(modes))
    
    print(f"  - 检查了 {len(img_files)} 张图像")
    print(f"  - 图像大小分布: {len(unique_sizes)} 种不同大小")
    if len(unique_sizes) <= 5:
        for size in unique_sizes:
            count = sizes.count(size)
            print(f"    {size}: {count} 张")
    else:
        # 显示最常见的几种
        from collections import Counter
        size_counts = Counter(sizes).most_common(5)
        for size, count in size_counts:
            print(f"    {size}: {count} 张")
    
    print(f"  - 颜色模式: {unique_modes}")


def main():
    # 配置数据路径
    data_root = 'data'  # 修改为你的数据路径
    csv_file = 'data/gt_training.csv'  # 修改为你的CSV路径
    
    # 检查当前目录
    print(f"当前工作目录: {os.getcwd()}\n")
    
    # 运行检查
    if check_data_structure(data_root, csv_file):
        check_image_properties(data_root)
        print("\n所有检查通过！可以开始训练了。")
    else:
        print("\n数据检查失败！请检查数据集。")


if __name__ == '__main__':
    main()
