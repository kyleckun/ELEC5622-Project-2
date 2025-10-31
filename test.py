import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import os

from model import get_model
from dataset import get_dataloaders

def test_model(model, test_loader, device):
    """测试模型"""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    print("\nTesting model...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return np.array(all_preds), np.array(all_labels), accuracy


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制混淆矩阵
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.title('Confusion Matrix (%)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to {save_path}")
    
    # 同时保存数值版本
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_accuracy(y_true, y_pred, class_names, save_path='results/class_accuracy.png'):
    """绘制每个类别的准确率柱状图"""
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_acc, color='steelblue', alpha=0.8)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{class_acc[i]:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Class', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Class accuracy plot saved to {save_path}")


def save_classification_report(y_true, y_pred, class_names, save_path='results/classification_report.txt'):
    """保存分类报告"""
    report = classification_report(y_true, y_pred, 
                                  target_names=class_names,
                                  digits=4)
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
        f.write(f"Overall Accuracy: {accuracy_score(y_true, y_pred)*100:.4f}%\n")
        f.write("="*60 + "\n")
    
    print(f"✓ Classification report saved to {save_path}")
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(report)
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred)*100:.4f}%")
    print("="*60)


def analyze_misclassifications(y_true, y_pred, class_names, top_n=3):
    """分析最常见的错误分类"""
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*60)
    print(f"Top {top_n} Misclassification Patterns:")
    print("="*60)
    
    # 找出非对角线元素（错误分类）
    misclass = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                misclass.append((cm[i, j], i, j))
    
    # 排序并显示top N
    misclass.sort(reverse=True)
    for idx, (count, true_idx, pred_idx) in enumerate(misclass[:top_n], 1):
        percentage = count / cm[true_idx].sum() * 100
        print(f"{idx}. {class_names[true_idx]} → {class_names[pred_idx]}: "
              f"{count} samples ({percentage:.2f}%)")
    print("="*60 + "\n")


def main():
    # ==================== 配置参数 ====================
    config = {
        'data_root': 'data',  # 数据根目录
        'csv_file': 'data/gt_training.csv',  # CSV文件路径
        'model_path': 'models/best_model.pth',  # 模型路径
        'results_dir': 'results',  # 结果保存目录
        'batch_size': 32,
    }
    
    # 类别名称
    class_names = ['Homogeneous', 'Speckled', 'Nucleolar', 
                   'Centromere', 'NuMem', 'Golgi']
    
    # 创建结果目录
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 加载数据
    print("Loading data...")
    _, _, test_loader = get_dataloaders(
        data_root=config['data_root'],
        csv_file=config['csv_file'],
        batch_size=config['batch_size'],
        augment=False  # 测试时不使用数据增强
    )
    
    # 加载模型
    print("Loading model...")
    model = get_model(num_classes=6, pretrained=False)
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ Model loaded from {config['model_path']}")
    if 'val_acc' in checkpoint:
        print(f"  Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # 测试
    y_pred, y_true, test_accuracy = test_model(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Total samples: {len(y_true)}")
    print(f"Correct predictions: {(y_pred == y_true).sum()}")
    print(f"{'='*60}\n")
    
    # 生成混淆矩阵
    plot_confusion_matrix(y_true, y_pred, class_names,
                         save_path=os.path.join(config['results_dir'], 'confusion_matrix.png'))
    
    # 生成每类准确率图
    plot_class_accuracy(y_true, y_pred, class_names,
                       save_path=os.path.join(config['results_dir'], 'class_accuracy.png'))
    
    # 保存分类报告
    save_classification_report(y_true, y_pred, class_names,
                              save_path=os.path.join(config['results_dir'], 'classification_report.txt'))
    
    # 分析错误分类
    analyze_misclassifications(y_true, y_pred, class_names, top_n=5)
    
    print(f"\n✓ All results saved to {config['results_dir']}/")


if __name__ == '__main__':
    main()
