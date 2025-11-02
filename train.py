import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import time
import warnings

# Filter torchvision image extension warning
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io.image')

from model import get_model
from dataset import get_dataloaders

class Trainer:
    def __init__(self, model, train_loader, val_loader, device,
                 lr=0.001, num_epochs=30, label_smoothing=0.1,
                 early_stopping_patience=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        # Loss function with Label Smoothing (防止过拟合)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"✓ Using Label Smoothing: {label_smoothing}")

        # Optimizer with moderate weight decay
        self.optimizer = optim.SGD(model.parameters(), lr=lr,
                                   momentum=0.9, weight_decay=5e-4)  # 适度的weight decay
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=10, gamma=0.1)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0  # Early stopping计数器
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validation"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, save_dir='models'):
        """Complete training process"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 60)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")

            # Save best model and early stopping check
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.patience_counter = 0  # 重置计数器
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
            else:
                self.patience_counter += 1
                print(f"  Early stopping patience: {self.patience_counter}/{self.early_stopping_patience}")

                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n{'='*60}")
                    print(f"Early stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                    print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
                    print(f"{'='*60}")
                    break

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"{'='*60}\n")

        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

        self.plot_history(save_dir)

        return total_time
    
    def plot_history(self, save_dir):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curves
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Learning rate curve
        axes[2].plot(self.history['lr'], linewidth=2, color='green')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Training curves saved to {save_dir}/training_curves.png")

    def save_experiment_log(self, config, total_time, experiment_name=None):
        """Save experiment results to log file"""
        from datetime import datetime

        # Generate experiment ID
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"

        # Load existing log
        log_file = 'experiments_log.json'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {}

        # Get final train accuracy from history
        final_train_acc = self.history['train_acc'][-1] if self.history['train_acc'] else 0

        # Create experiment entry
        experiment_entry = {
            "experiment_id": experiment_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "model": "AlexNet",
                "freeze_features": config.get('freeze_features', True),
                "batch_size": config.get('batch_size', 16),
                "lr": config.get('lr', 0.0005),
                "dropout_p": config.get('dropout_p', 0.7),
                "label_smoothing": config.get('label_smoothing', 0.1),
                "weight_decay": 0.001,
                "lr_schedule": "StepLR(step_size=10, gamma=0.1)",
                "early_stopping_patience": config.get('early_stopping_patience', 15),
                "augmentation": "strong (8 techniques)" if config.get('augment', True) else "basic"
            },
            "results": {
                "best_epoch": self.best_epoch,
                "total_epochs": len(self.history['train_loss']),
                "train_acc": round(final_train_acc, 2),
                "val_acc": round(self.best_val_acc, 2),
                "test_acc": None,  # To be filled after testing
                "training_time_min": round(total_time / 60, 2)
            },
            "notes": ""
        }

        # Add to log
        log_data[experiment_name] = experiment_entry

        # Save log
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n✓ Experiment logged: {experiment_name}")
        print(f"  Val Acc: {self.best_val_acc:.2f}%")
        print(f"  See experiments_log.json for details")


def main():
    # Configuration parameters - AGGRESSIVE VERSION (追赶同学)
    config = {
        'data_root': 'data',  # Data root directory (contains train/val/test)
        'csv_file': 'data/gt_training.csv',  # CSV file path
        'batch_size': 32,  # 增大batch size，更稳定的梯度
        'lr': 0.001,  # 提高学习率，加快收敛
        'num_epochs': 50,  # 最大epoch数
        'augment': True,  # Whether to use data augmentation
        'freeze_features': False,  # 不冻结！允许微调整个网络
        'dropout_p': 0.5,  # 降低dropout，增强学习能力
        'label_smoothing': 0.0,  # 去掉label smoothing
        'early_stopping_patience': 15,  # Early stopping耐心值
    }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=config['data_root'],
        csv_file=config['csv_file'],
        batch_size=config['batch_size'],
        augment=config['augment']
    )

    print(f"\n✓ Data loaded successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    print("\n" + "="*60)
    print("Configuration Summary:")
    print("="*60)
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Dropout: {config['dropout_p']}")
    print(f"  Label smoothing: {config['label_smoothing']}")
    print(f"  Freeze features: {config['freeze_features']}")
    print(f"  Early stopping patience: {config['early_stopping_patience']}")
    print("="*60)

    print("\nCreating model...")
    model = get_model(num_classes=6, pretrained=True,
                     freeze_features=config['freeze_features'],
                     dropout_p=config['dropout_p'])

    trainer = Trainer(model, train_loader, val_loader, device,
                     lr=config['lr'], num_epochs=config['num_epochs'],
                     label_smoothing=config['label_smoothing'],
                     early_stopping_patience=config['early_stopping_patience'])

    # Train and log experiment
    total_time = trainer.train(save_dir='models')

    # Ask for experiment name (optional)
    print("\n" + "="*60)
    exp_name = input("Enter experiment name (or press Enter for auto-generated): ").strip()
    if not exp_name:
        exp_name = None

    trainer.save_experiment_log(config, total_time, experiment_name=exp_name)


if __name__ == '__main__':
    main()
