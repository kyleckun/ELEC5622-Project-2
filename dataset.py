import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# Filter torchvision image extension warning
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io.image')

from torchvision import transforms

class HEp2Dataset(Dataset):
    """HEp-2 Cell Image Dataset - 适配ICPR2014格式"""
    
    def __init__(self, img_dir, csv_file, transform=None):
        """
        Args:
            img_dir: 图像文件夹路径 (train/val/test)
            csv_file: CSV标签文件路径
            transform: 数据增强transforms
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 创建类别到数字的映射
        self.class_to_idx = {
            'Homogeneous': 0,
            'Speckled': 1,
            'Nucleolar': 2,
            'Centromere': 3,
            'NuMem': 4,
            'Golgi': 5
        }
        
        # 提取当前目录的图像
        self.samples = []
        
        # 获取当前文件夹中的所有图像文件
        img_files = set([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        print(f"Found {len(img_files)} images in {img_dir}")
        
        # 遍历CSV，匹配当前文件夹的图像
        for idx, row in df.iterrows():
            img_id = row['Image ID']
            img_class = row['Image class']
            
            # 转换ID到文件名：1 -> 00001.png
            img_filename = f"{img_id:05d}.png"
            
            # 检查文件是否在当前目录
            if img_filename in img_files:
                img_path = os.path.join(img_dir, img_filename)
                label = self.class_to_idx[img_class]
                self.samples.append((img_path, label))
        
        print(f"Loaded {len(self.samples)} samples from {img_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms(augment=True):
    """
    获取数据预处理transforms

    Args:
        augment: 是否使用数据增强
    """
    if augment:
        # 训练集：使用强数据增强（防止过拟合）
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # 随机裁剪+缩放
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # 增加垂直翻转
            transforms.RandomRotation(30),  # 增加旋转角度
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移
            transforms.ColorJitter(
                brightness=0.3,  # 增加亮度变化
                contrast=0.3,    # 增加对比度变化
                saturation=0.3,  # 增加饱和度变化
                hue=0.15         # 增加色调变化
            ),
            transforms.RandomGrayscale(p=0.1),  # 10%概率转灰度
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),  # Random Erasing
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        print("✓ Using strong data augmentation for training")
    else:
        # 不使用增强的baseline
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    # 验证集和测试集：不使用数据增强
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def get_dataloaders(data_root, csv_file, batch_size=32, augment=True):
    """
    创建训练、验证、测试的DataLoader
    
    Args:
        data_root: 数据根目录（包含train/val/test文件夹）
        csv_file: CSV标签文件路径
        batch_size: batch大小
        augment: 是否使用数据增强
    """
    train_transform, val_transform = get_data_transforms(augment)
    
    # 创建数据集
    train_dataset = HEp2Dataset(
        img_dir=os.path.join(data_root, 'train'),
        csv_file=csv_file,
        transform=train_transform
    )
    
    val_dataset = HEp2Dataset(
        img_dir=os.path.join(data_root, 'val'),
        csv_file=csv_file,
        transform=val_transform
    )
    
    test_dataset = HEp2Dataset(
        img_dir=os.path.join(data_root, 'test'),
        csv_file=csv_file,
        transform=val_transform
    )
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader


# 测试代码
if __name__ == '__main__':
    # 测试数据加载
    data_root = 'path/to/your/data'  # 替换为你的数据路径
    csv_file = 'path/to/gt_training.csv'  # 替换为CSV路径
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=data_root,
        csv_file=csv_file,
        batch_size=32,
        augment=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 测试加载一个batch
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(f"Labels: {labels}")
        break
