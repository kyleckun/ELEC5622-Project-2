import torch
import torch.nn as nn
from torchvision import models

class AlexNetFinetune(nn.Module):
    """
    Fine-tuned AlexNet for HEp-2 cell classification
    """
    
    def __init__(self, num_classes=6, pretrained=True):
        super(AlexNetFinetune, self).__init__()
        
        # 加载预训练的AlexNet
        self.model = models.alexnet(pretrained=pretrained)
        
        # 修改最后的全连接层：1000 -> num_classes
        # AlexNet的分类器结构：
        # (classifier): Sequential(
        #   (0): Dropout(p=0.5)
        #   (1): Linear(in_features=9216, out_features=4096)
        #   (2): ReLU(inplace=True)
        #   (3): Dropout(p=0.5)
        #   (4): Linear(in_features=4096, out_features=4096)
        #   (5): ReLU(inplace=True)
        #   (6): Linear(in_features=4096, out_features=1000)
        # )
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_features(self):
        """冻结特征提取层，只训练分类器"""
        for param in self.model.features.parameters():
            param.requires_grad = False
        print("✓ Feature layers frozen. Only training classifier.")
    
    def unfreeze_all(self):
        """解冻所有层"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("✓ All layers unfrozen.")


def get_model(num_classes=6, pretrained=True, freeze_features=False):
    """
    创建并返回模型
    
    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        freeze_features: 是否冻结特征提取层
    """
    model = AlexNetFinetune(num_classes=num_classes, pretrained=pretrained)
    
    if freeze_features:
        model.freeze_features()
    else:
        print("✓ All layers will be trained.")
    
    return model


# 测试代码
if __name__ == '__main__':
    # 测试模型创建
    model = get_model(num_classes=6, pretrained=True, freeze_features=False)
    
    # 打印模型结构
    print("\n" + "="*50)
    print("Model Architecture:")
    print("="*50)
    print(model)
    
    # 测试前向传播
    print("\n" + "="*50)
    print("Testing forward pass...")
    print("="*50)
    x = torch.randn(2, 3, 224, 224)  # batch=2, channels=3, H=224, W=224
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Model works correctly!")
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
