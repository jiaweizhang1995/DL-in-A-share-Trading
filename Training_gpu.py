import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import time
import os
import logging
import argparse
import json
from datetime import datetime, timedelta
import gc

if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_gpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 命令行参数解析
parser = argparse.ArgumentParser(description='GPU Training Script with Optimization')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from')
parser.add_argument('--checkpoint-freq', type=int, default=5, help='Save checkpoint every N epochs')
parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training (larger for GPU)')
parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID to use')
parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training (FP16)')
parser.add_argument('--num-workers', type=int, default=8, help='Number of data loading workers')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
args = parser.parse_args()

# GPU设备检查和设置
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(args.gpu_id)
    logger.info(f"使用GPU设备: {device}")
    logger.info(f"GPU名称: {torch.cuda.get_device_name(args.gpu_id)}")
    logger.info(f"GPU内存: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3:.1f} GB")
    # 设置CUDA优化选项
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    device = torch.device("cpu")
    logger.warning("CUDA不可用，使用CPU设备")

logger.info(f"开始训练流程 (GPU优化版本) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"批次大小: {args.batch_size}, 数据加载线程数: {args.num_workers}")
logger.info(f"混合精度训练: {'启用' if args.mixed_precision else '禁用'}")

# 混合精度训练设置
if args.mixed_precision and torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    logger.info("启用混合精度训练 (FP16)")
else:
    args.mixed_precision = False
    logger.info("使用标准精度训练 (FP32)")

if args.resume:
    logger.info(f"将从 checkpoint 恢复训练: {args.resume}")
else:
    logger.info("从头开始训练")

# Checkpoint 保存和加载函数
def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, aucs, best_corr, best_epoch, filename):
    """保存训练 checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'aucs': aucs,
        'best_corr': best_corr,
        'best_epoch': best_epoch,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args)
    }
    if args.mixed_precision:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint 已保存: {filename}")

def load_checkpoint(filename, model, optimizer, scheduler):
    """加载训练 checkpoint"""
    if os.path.isfile(filename):
        logger.info(f"加载 checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=device)
        
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if args.mixed_precision and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        aucs = checkpoint['aucs']
        best_corr = checkpoint['best_corr']
        best_epoch = checkpoint['best_epoch']
        
        logger.info(f"恢复训练，从 epoch {start_epoch} 开始")
        logger.info(f"当前最佳相关系数: {best_corr:.6f} (Epoch {best_epoch+1})")
        
        return start_epoch, train_losses, val_losses, aucs, best_corr, best_epoch
    else:
        logger.error(f"Checkpoint 文件不存在: {filename}")
        return None

# 如果恢复训练，检查是否存在预处理的数据
if args.resume and os.path.exists('./processed_data.npz'):
    logger.info("从缓存加载预处理数据...")
    data = np.load('./processed_data.npz')
    train_data_scaled = data['train_data']
    val_data_scaled = data['val_data']
    train_label_scaled = data['train_label']
    val_label_scaled = data['val_label']
    logger.info(f"加载完成 - 训练集: {train_data_scaled.shape}, 验证集: {val_data_scaled.shape}")
else:
    logger.info("开始数据加载和预处理...")
    df = pd.read_csv('./data/20250701.csv')
    logger.info(f"原始数据加载完成: {df.shape[0]} 条记录")

    df['amount'] = df['amount'] / 10000000
    original_stocks = df['stock_id'].nunique()
    df = df.groupby('stock_id').filter(lambda x: len(x) >= 180)  # 少于180交易日的股票不要
    df = df.groupby('stock_id').apply(lambda x: x.iloc[20:], include_groups=False)  # 刚上市的20个交易日不要
    df.reset_index(inplace=True)
    df.drop('level_1', axis=1, inplace=True)
    logger.info(f"过滤后保留 {df['stock_id'].nunique()}/{original_stocks} 只股票")

    df['stock_id'] = df['stock_id'].astype(str)
    df = df[~df['stock_id'].str.startswith('8')]
    df = df[~df['stock_id'].str.startswith('68')]
    df = df[~df['stock_id'].str.startswith('4')]
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"最终处理后: {df['stock_id'].nunique()} 只股票, {df.shape[0]} 条记录")

    # 训练集和测试集分割
    train_df = df[df['date'] <= pd.to_datetime('2021-12-31')]
    val_df = df[df['date'] > pd.to_datetime('2021-12-31')]
    logger.info(f"训练集时间范围: {train_df['date'].min()} 到 {train_df['date'].max()}")
    logger.info(f"验证集时间范围: {val_df['date'].min()} 到 {val_df['date'].max()}")

    logger.info("开始处理训练数据样本...")
    data_start_time = time.time()

    def process_data(df, desc="处理数据"):
        grouped = df.groupby('stock_id')
        samples = []
        labels = []
        total_groups = len(grouped)
        
        for idx, (stock_id, group) in enumerate(tqdm(grouped, desc=desc)):
            product_samples = group.values
            num_samples = len(product_samples)
            if num_samples < 180:
                continue
            
            if (idx + 1) % 100 == 0:
                logger.info(f"已处理 {idx + 1}/{total_groups} 只股票")
            
            for i in range(num_samples - 85):
                try:
                    LLL = product_samples[i:i + 86, 2:6].astype(np.float32)
                    LLLL = product_samples[i:i + 86, 8:9].astype(np.float32)
                    LLLLL = product_samples[i:i + 86, 9:10].astype(np.float32)
                except (ValueError, TypeError):
                    continue
                
                if (np.any(np.isnan(LLL)) or np.any(np.isnan(LLLL)) or np.any(np.isnan(LLLLL))):
                    continue
                if np.any(LLL <= 0):
                    continue
                if np.any(LLLL < 0) or np.any(LLLL > 50):
                    continue
                if np.any(LLLLL < -50) or np.any(LLLLL > 50):
                    continue
                
                sample = product_samples[i:i + 60, 2:]
                
                try:
                    l = float(product_samples[i + 60, 2])
                    Hl = float(product_samples[i + 60, 3])
                    if l != Hl:
                        volume_data = product_samples[i + 61:i + 86, 6].astype(np.float32)
                        ll = np.mean(volume_data[~np.isnan(volume_data)])
                        if not np.isnan(ll) and ll > 0 and l > 0:
                            lll = ((ll - l) / l) * 100
                            if not np.isnan(lll):
                                labels.append(float(lll))
                                samples.append(sample[:, [0, 1, 2, 3, 5, -1]])
                except (ValueError, TypeError, IndexError):
                    continue
        
        return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.float32)

    train_data, train_label = process_data(train_df, "处理训练数据")
    val_data, val_label = process_data(val_df, "处理验证数据")

    data_process_time = time.time() - data_start_time
    logger.info(f"数据处理完成，耗时: {data_process_time/60:.1f} 分钟")
    logger.info(f"训练集: {train_data.shape}, 验证集: {val_data.shape}")

    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    
    logger.info("开始数据标准化...")
    scaler_features = StandardScaler()
    scaler_labels = StandardScaler()

    # 标准化训练数据
    original_shape = train_data.shape
    train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
    train_data_scaled = scaler_features.fit_transform(train_data_reshaped)
    train_data_scaled = train_data_scaled.reshape(original_shape).astype(np.float32)

    train_label_scaled = scaler_labels.fit_transform(train_label.reshape(-1, 1)).flatten().astype(np.float32)

    # 标准化验证数据
    val_original_shape = val_data.shape
    val_data_reshaped = val_data.reshape(-1, val_data.shape[-1])
    val_data_scaled = scaler_features.transform(val_data_reshaped)
    val_data_scaled = val_data_scaled.reshape(val_original_shape).astype(np.float32)

    val_label_scaled = scaler_labels.transform(val_label.reshape(-1, 1)).flatten().astype(np.float32)

    logger.info("数据标准化完成")

    # 保存预处理数据以便后续恢复训练
    np.savez('./processed_data.npz', 
             train_data=train_data_scaled, val_data=val_data_scaled,
             train_label=train_label_scaled, val_label=val_label_scaled)
    logger.info("预处理数据已保存到 processed_data.npz")

    # 释放原始数据内存
    del train_data, val_data, train_label, val_label, df, train_df, val_df
    gc.collect()

# GPU优化的神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(60 * 6, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100, 1)
        
        # 1D convolution layers
        self.conv1d_1 = nn.Conv1d(6, 16, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        
        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        
        # LSTM layers
        self.lstm = nn.LSTM(6, 64, 4, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Batch normalization and dropout
        self.bn1 = nn.BatchNorm1d(2000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout(0.3)
        
        # Feature combination layer
        self.feature_combine = nn.Linear(2000 + 32 + 32 + 128, 200)
        self.final_output = nn.Linear(200, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Min-max normalization
        x_min = x.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        x_max = x.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        x_normalized = (x - x_min) / (x_max - x_min + 1e-8)
        
        # Fully connected path
        fc_input = x_normalized.view(batch_size, -1)
        fc_out = torch.relu(self.bn1(self.fc1(fc_input)))
        fc_out = self.dropout(fc_out)
        fc_out = torch.relu(self.bn2(self.fc2(fc_out)))
        fc_out = self.dropout(fc_out)
        fc_out = torch.relu(self.bn3(self.fc3(fc_out)))
        
        # 1D convolution path
        conv1d_input = x_normalized.permute(0, 2, 1)
        conv1d_out = torch.relu(self.conv1d_1(conv1d_input))
        conv1d_out = torch.relu(self.conv1d_2(conv1d_out))
        conv1d_out = self.pool1d(conv1d_out).view(batch_size, -1)
        
        # 2D convolution path
        conv2d_input = x_normalized.unsqueeze(1)
        conv2d_out = torch.relu(self.conv2d_1(conv2d_input))
        conv2d_out = torch.relu(self.conv2d_2(conv2d_out))
        conv2d_out = self.pool2d(conv2d_out).view(batch_size, -1)
        
        # LSTM path
        lstm_out, _ = self.lstm(x_normalized)
        lstm_out = lstm_out[:, -1, :]
        
        # Combine all features
        combined_features = torch.cat([fc_out, conv1d_out, conv2d_out, lstm_out], dim=1)
        combined_out = torch.relu(self.feature_combine(combined_features))
        combined_out = self.dropout(combined_out)
        
        output = self.final_output(combined_out)
        return output

# GPU优化的数据集类
class StockDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx:idx+1]

# 创建数据集和数据加载器
train_dataset = StockDataset(train_data_scaled, train_label_scaled)
val_dataset = StockDataset(val_data_scaled, val_label_scaled)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                         num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                       num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

logger.info(f"数据加载器创建完成 - 训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

# 初始化模型
model = NeuralNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

# 初始化训练变量
start_epoch = 0
train_losses = []
val_losses = []
aucs = []
best_corr = -float('inf')
best_epoch = 0

# 如果恢复训练，加载checkpoint
if args.resume:
    checkpoint_data = load_checkpoint(args.resume, model, optimizer, scheduler)
    if checkpoint_data:
        start_epoch, train_losses, val_losses, aucs, best_corr, best_epoch = checkpoint_data

num_epochs = args.epochs
logger.info(f"开始训练，总轮数: {num_epochs}, 从第 {start_epoch+1} 轮开始")

for epoch in range(start_epoch, num_epochs):
    epoch_start_time = time.time()
    
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    
    for batch_data, batch_labels in train_progress:
        batch_data, batch_labels = batch_data.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if args.mixed_precision:
            with autocast():
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item()
        train_progress.set_postfix({'Loss': f'{loss.item():.6f}'})
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        for batch_data, batch_labels in val_progress:
            batch_data, batch_labels = batch_data.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
            
            if args.mixed_precision:
                with autocast():
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
            else:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
            
            val_loss += loss.item()
            
            val_predictions.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(batch_labels.cpu().numpy().flatten())
            val_progress.set_postfix({'Loss': f'{loss.item():.6f}'})
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # 计算相关系数
    correlation = np.corrcoef(val_predictions, val_targets)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    aucs.append(correlation)
    
    # 更新学习率
    scheduler.step(avg_val_loss)
    
    epoch_time = time.time() - epoch_start_time
    logger.info(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, '
               f'Correlation: {correlation:.6f}, Time: {epoch_time:.1f}s')
    
    # 保存最佳模型
    if correlation > best_corr:
        best_corr = correlation
        best_epoch = epoch
        torch.save(model.state_dict(), './weights/model_baseline.pt')
        logger.info(f'保存最佳模型，相关系数: {best_corr:.6f}')
    
    # 定期保存checkpoint
    if (epoch + 1) % args.checkpoint_freq == 0:
        checkpoint_path = f'./checkpoints/training_checkpoint_epoch_{epoch}.pt'
        save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, 
                       aucs, best_corr, best_epoch, checkpoint_path)
    
    # GPU内存管理
    if epoch % 10 == 0:
        torch.cuda.empty_cache()
        gc.collect()

logger.info(f"训练完成！最佳相关系数: {best_corr:.6f} (Epoch {best_epoch+1})")

# 绘制训练曲线
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(aucs, label='Validation Correlation', color='green')
plt.title('Validation Correlation')
plt.xlabel('Epoch')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(val_targets, val_predictions, alpha=0.5)
plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--')
plt.title(f'Test Set Correlation: {correlation:.4f}')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)

plt.tight_layout()
plt.savefig('baseline_training_results_gpu.png', dpi=300, bbox_inches='tight')
logger.info("训练结果图表已保存到 baseline_training_results_gpu.png")

# 保存训练结果
results = {
    'best_correlation': float(best_corr),
    'best_epoch': int(best_epoch + 1),
    'final_train_loss': float(train_losses[-1]),
    'final_val_loss': float(val_losses[-1]),
    'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_epochs': num_epochs,
    'batch_size': args.batch_size,
    'mixed_precision': args.mixed_precision,
    'gpu_name': torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else 'CPU'
}

with open('./weights/training_results_gpu.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

logger.info("训练结果已保存到 ./weights/training_results_gpu.json")

# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    logger.info("GPU内存已清理")

logger.info("训练流程全部完成！")