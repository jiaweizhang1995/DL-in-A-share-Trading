import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import time
from tqdm import tqdm
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
        logging.FileHandler('5_fold_cv_gpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 命令行参数解析
parser = argparse.ArgumentParser(description='5-Fold Cross-Validation GPU Optimized Version')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from')
parser.add_argument('--checkpoint-freq', type=int, default=5, help='Save checkpoint every N epochs')
parser.add_argument('--resume-fold', type=int, default=None, help='Resume from specific fold (0-based)')
parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training (larger for GPU)')
parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID to use')
parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training (FP16)')
parser.add_argument('--num-workers', type=int, default=8, help='Number of data loading workers')
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

logger.info(f"开始5折交叉验证 (GPU优化版本) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

# Checkpoint 保存和加载函数
def save_checkpoint(fold, epoch, model, optimizer, scheduler, train_losses, val_losses, aucs, best_corr, best_epoch, all_fold_results, filename):
    """保存交叉验证 checkpoint"""
    checkpoint = {
        'fold': fold,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'aucs': aucs,
        'best_corr': best_corr,
        'best_epoch': best_epoch,
        'all_fold_results': all_fold_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args)
    }
    if args.mixed_precision:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint 已保存: {filename}")

def load_checkpoint(filename):
    """加载交叉验证 checkpoint"""
    if os.path.isfile(filename):
        logger.info(f"加载 checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=device)
        
        fold = checkpoint['fold']
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        aucs = checkpoint['aucs']
        best_corr = checkpoint['best_corr']
        best_epoch = checkpoint['best_epoch']
        all_fold_results = checkpoint['all_fold_results']
        
        if args.mixed_precision and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"恢复训练，从 fold {fold+1}, epoch {start_epoch} 开始")
        logger.info(f"当前最佳相关系数: {best_corr:.6f} (Epoch {best_epoch+1})")
        
        return fold, start_epoch, train_losses, val_losses, aucs, best_corr, best_epoch, all_fold_results, checkpoint
    else:
        logger.error(f"Checkpoint 文件不存在: {filename}")
        return None

logger.info("开始加载和预处理数据...")
df = pd.read_csv('./data/20250701.csv')
logger.info(f"原始数据加载完成: {df.shape[0]} 条记录")

df['amount'] = df['amount'] / 10000000
original_stocks = df['stock_id'].nunique()
df = df.groupby('stock_id').filter(lambda x: len(x) >= 180)  # 少于180交易日的股票不要
df = df.groupby('stock_id').apply(lambda x: x.iloc[20:], include_groups=False)  # 刚上市的20个交易日不要
df.reset_index(inplace=True)  # 重置索引，将stock_id从索引恢复为列
df.drop('level_1', axis=1, inplace=True)  # 删除多余的level_1列
logger.info(f"过滤后保留 {df['stock_id'].nunique()}/{original_stocks} 只股票")

df['stock_id'] = df['stock_id'].astype(str)
df = df[~df['stock_id'].str.startswith('8')]
df = df[~df['stock_id'].str.startswith('68')]
df = df[~df['stock_id'].str.startswith('4')]
df['date'] = pd.to_datetime(df['date'])
# 使用与Training.py相同的训练数据时间范围
df = df[(df['date'] > pd.to_datetime('2010-01-01')) & (df['date'] < pd.to_datetime('2024-01-01'))]
logger.info(f"最终处理后: {df['stock_id'].nunique()} 只股票, {df.shape[0]} 条记录")

logger.info("开始处理训练数据样本...")
data_start_time = time.time()
grouped = df.groupby('stock_id')
samples = []
label = []
total_groups = len(grouped)
logger.info(f"需要处理 {total_groups} 只股票的数据")

for idx, (stock_id, group) in enumerate(tqdm(grouped, desc="处理训练数据")):
    product_samples = group.values
    num_samples = len(product_samples)
    if num_samples < 180:
        continue
    
    if (idx + 1) % 100 == 0:
        elapsed = time.time() - data_start_time
        eta = elapsed * (total_groups / (idx + 1) - 1)
        logger.info(f"已处理 {idx + 1}/{total_groups} 只股票, 预计剩余时间: {eta/60:.1f} 分钟")
    
    for i in range(num_samples - 85):
        try:
            LLL = product_samples[i:i + 86, 2:6].astype(np.float32)  # open, close, high, low
            LLLL = product_samples[i:i + 86, 8:9].astype(np.float32)  # turn
            LLLLL = product_samples[i:i + 86, 9:10].astype(np.float32)  # pctChg
        except (ValueError, TypeError):
            # 转换失败，跳过这个样本
            continue
        
        # 跳过包含NaN的样本
        if (np.any(np.isnan(LLL)) or np.any(np.isnan(LLLL)) or np.any(np.isnan(LLLLL))):
            continue
            
        # 检查价格是否合理（大于0）
        if np.any(LLL <= 0):
            continue
            
        # 检查换手率是否合理
        if np.any(LLLL < 0) or np.any(LLLL > 50):  # 放宽换手率限制
            continue
            
        # 检查涨跌幅是否合理  
        if np.any(LLLLL < -50) or np.any(LLLLL > 50):  # 放宽涨跌幅限制
            continue
        
        sample = product_samples[i:i + 60, 2:]  # 从open开始到最后一列

        try:
            l = float(product_samples[i + 60, 2])  # open
            Hl = float(product_samples[i + 60, 3])  # close
            if l != Hl:
                volume_data = product_samples[i + 61:i + 86, 6].astype(np.float32)  # volume
                ll = np.mean(volume_data[~np.isnan(volume_data)])  # 忽略NaN值计算平均
                if not np.isnan(ll) and ll > 0 and l > 0:  # 确保volume和价格有效
                    lll = ((ll - l) / l)*100
                    if not np.isnan(lll):  # 确保标签有效
                        label.append(float(lll))
                        samples.append(sample[ :, [0, 1, 2, 3, 5, -1]]) #开、收、高、低、成交额、换手率
        except (ValueError, TypeError, IndexError):
            # 数据转换失败，跳过
            continue

train_data = np.array(samples)
train_data = train_data.astype(np.float32)
train_label = np.array(label).astype(np.float32)
data_process_time = time.time() - data_start_time
logger.info(f"训练数据处理完成: {train_data.shape}, 耗时: {data_process_time/60:.1f} 分钟")
logger.info(f"训练样本数: {len(samples)}, 标签数: {len(label)}")

# 数据标准化防止数值不稳定
from sklearn.preprocessing import StandardScaler

logger.info("开始数据标准化...")
# 标准化训练数据
original_shape = train_data.shape
scaler_features = StandardScaler()
scaler_labels = StandardScaler()

# 重塑数据进行标准化
train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
train_data_scaled = scaler_features.fit_transform(train_data_reshaped)
train_data_scaled = train_data_scaled.reshape(original_shape)

train_label_scaled = scaler_labels.fit_transform(train_label.reshape(-1, 1)).flatten()

logger.info("数据标准化完成")

# 释放原始数据内存
del samples, label, train_data, train_label
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

# 5折交叉验证
kfold = TimeSeriesSplit(n_splits=5)
all_fold_results = []

# 检查是否需要恢复训练
start_fold = 0
if args.resume:
    resume_data = load_checkpoint(args.resume)
    if resume_data:
        start_fold, start_epoch, train_losses, val_losses, aucs, best_corr, best_epoch, all_fold_results, checkpoint = resume_data

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data_scaled)):
    if fold < start_fold:
        continue
        
    logger.info(f"\n开始第 {fold+1}/5 折训练")
    
    # 创建数据集
    X_train_fold = train_data_scaled[train_idx]
    y_train_fold = train_label_scaled[train_idx]
    X_val_fold = train_data_scaled[val_idx]
    y_val_fold = train_label_scaled[val_idx]
    
    # 创建数据加载器，使用更多workers提高GPU利用率
    train_dataset = StockDataset(X_train_fold, y_train_fold)
    val_dataset = StockDataset(X_val_fold, y_val_fold)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    
    # 初始化模型
    model = NeuralNetwork().to(device)
    
    # GPU优化的优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
    
    # 如果恢复训练，加载模型状态
    if args.resume and fold == start_fold:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch_for_fold = start_epoch
    else:
        best_corr = -float('inf')
        best_epoch = 0
        train_losses = []
        val_losses = []
        aucs = []
        start_epoch_for_fold = 0
    
    num_epochs = 50
    
    for epoch in range(start_epoch_for_fold, num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{num_epochs} [Train]')
        
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
            val_progress = tqdm(val_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{num_epochs} [Val]')
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
        
        logger.info(f'Fold {fold+1} Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Correlation: {correlation:.6f}')
        
        # 保存最佳模型
        if correlation > best_corr:
            best_corr = correlation
            best_epoch = epoch
            torch.save(model.state_dict(), f'./weights/model_APP_{fold}.pt')
            logger.info(f'保存第 {fold+1} 折最佳模型，相关系数: {best_corr:.6f}')
        
        # 定期保存checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = f'./checkpoints/cv_checkpoint_fold_{fold}_epoch_{epoch}.pt'
            save_checkpoint(fold, epoch, model, optimizer, scheduler, train_losses, val_losses, 
                          aucs, best_corr, best_epoch, all_fold_results, checkpoint_path)
        
        # GPU内存管理
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # 记录本折结果
    fold_result = {
        'fold': fold + 1,
        'best_correlation': best_corr,
        'best_epoch': best_epoch + 1,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }
    all_fold_results.append(fold_result)
    
    logger.info(f'第 {fold+1} 折训练完成，最佳相关系数: {best_corr:.6f} (Epoch {best_epoch+1})')
    
    # 清理内存
    del model, optimizer, train_dataset, val_dataset, train_loader, val_loader
    del X_train_fold, y_train_fold, X_val_fold, y_val_fold
    torch.cuda.empty_cache()
    gc.collect()

# 打印所有折的结果
logger.info("\n=== 5折交叉验证结果汇总 ===")
correlations = [result['best_correlation'] for result in all_fold_results]
mean_corr = np.mean(correlations)
std_corr = np.std(correlations)

for result in all_fold_results:
    logger.info(f"第 {result['fold']} 折: 最佳相关系数 = {result['best_correlation']:.6f} (Epoch {result['best_epoch']})")

logger.info(f"\n平均相关系数: {mean_corr:.6f} ± {std_corr:.6f}")
logger.info(f"最佳单折相关系数: {max(correlations):.6f}")
logger.info(f"交叉验证完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 保存结果
results_summary = {
    'cross_validation_results': all_fold_results,
    'mean_correlation': float(mean_corr),
    'std_correlation': float(std_corr),
    'best_correlation': float(max(correlations)),
    'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'batch_size': args.batch_size,
    'mixed_precision': args.mixed_precision,
    'gpu_name': torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else 'CPU'
}

with open('./weights/cv_results_gpu.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

logger.info("结果已保存到 ./weights/cv_results_gpu.json")

# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    logger.info("GPU内存已清理")