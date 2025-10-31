import torch
import torch.nn as nn
from torchvision import models

class AlexNetFinetune(nn.Module):
    """
    Fine-tuned AlexNet for HEp-2 cell classification
    """
    
    def __init__(self, num_classes=6, pretrained=True, dropout_p=0.7):
        super(AlexNetFinetune, self).__init__()

        # 加载预训练的AlexNet（使用新的weights参数）
        if pretrained:
            from torchvision.models import AlexNet_Weights
            self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        else:
            self.model = models.alexnet(weights=None)

        # 修改分类器以增强正则化（防止过拟合）
        # 增加Dropout比率：0.5 -> 0.7
        # 减少中间层神经元数量：4096 -> 2048
        num_features = self.model.classifier[1].in_features  # 9216

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, num_classes)
        )

        print(f"✓ Modified classifier with Dropout={dropout_p}, reduced neurons")
        
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


def get_model(num_classes=6, pretrained=True, freeze_features=False, dropout_p=0.7):
    """
    创建并返回模型

    Args:
        num_classes: 类别数量
        pretrained: 是否使用预训练权重
        freeze_features: 是否冻结特征提取层
        dropout_p: Dropout概率（默认0.7，增强正则化）
    """
    model = AlexNetFinetune(num_classes=num_classes, pretrained=pretrained, dropout_p=dropout_p)

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