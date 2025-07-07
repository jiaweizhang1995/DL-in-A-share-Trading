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
import gc  # 垃圾回收

if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('5_fold_cv_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 命令行参数解析
parser = argparse.ArgumentParser(description='5-Fold Cross-Validation with Memory Optimization')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from')
parser.add_argument('--checkpoint-freq', type=int, default=5, help='Save checkpoint every N epochs')
parser.add_argument('--resume-fold', type=int, default=None, help='Resume from specific fold (0-based)')
parser.add_argument('--sample-ratio', type=float, default=0.3, help='Sample ratio to reduce memory usage (0.1-1.0)')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")
logger.info(f"开始5折交叉验证 (内存优化版本) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"采样比例: {args.sample_ratio}, 批次大小: {args.batch_size}")

# 内存优化的数据加载函数
def load_and_sample_data(sample_ratio=0.3):
    """加载数据并进行采样以减少内存使用"""
    logger.info("开始加载和预处理数据...")
    df = pd.read_csv('./data/20250701.csv')
    logger.info(f"原始数据加载完成: {df.shape[0]} 条记录")

    df['amount'] = df['amount'] / 10000000
    original_stocks = df['stock_id'].nunique()
    
    # 更严格的筛选条件以减少数据量
    df = df.groupby('stock_id').filter(lambda x: len(x) >= 250)  # 增加到250交易日
    df = df.groupby('stock_id').apply(lambda x: x.iloc[30:], include_groups=False)  # 增加到30个交易日
    df.reset_index(inplace=True)
    df.drop('level_1', axis=1, inplace=True)
    logger.info(f"过滤后保留 {df['stock_id'].nunique()}/{original_stocks} 只股票")

    df['stock_id'] = df['stock_id'].astype(str)
    df = df[~df['stock_id'].str.startswith('8')]
    df = df[~df['stock_id'].str.startswith('68')]
    df = df[~df['stock_id'].str.startswith('4')]
    df['date'] = pd.to_datetime(df['date'])
    
    # 限制时间范围以减少数据量
    df = df[(df['date'] > pd.to_datetime('2015-01-01')) & (df['date'] < pd.to_datetime('2024-01-01'))]
    
    # 随机采样股票以减少数据量
    if sample_ratio < 1.0:
        stock_ids = df['stock_id'].unique()
        sample_size = int(len(stock_ids) * sample_ratio)
        sampled_stocks = np.random.choice(stock_ids, size=sample_size, replace=False)
        df = df[df['stock_id'].isin(sampled_stocks)]
        logger.info(f"采样后保留 {len(sampled_stocks)} 只股票")
    
    logger.info(f"最终处理后: {df['stock_id'].nunique()} 只股票, {df.shape[0]} 条记录")
    return df

# 内存优化的数据处理函数
def process_data_batched(df, batch_size=100):
    """分批处理数据以减少内存使用"""
    logger.info("开始分批处理训练数据样本...")
    data_start_time = time.time()
    grouped = df.groupby('stock_id')
    
    all_samples = []
    all_labels = []
    total_groups = len(grouped)
    logger.info(f"需要处理 {total_groups} 只股票的数据")
    
    # 分批处理
    groups_list = list(grouped)
    for batch_start in range(0, total_groups, batch_size):
        batch_end = min(batch_start + batch_size, total_groups)
        batch_samples = []
        batch_labels = []
        
        for idx in range(batch_start, batch_end):
            stock_id, group = groups_list[idx]
            product_samples = group.values
            num_samples = len(product_samples)
            
            if num_samples < 180:
                continue
            
            # 对每只股票的样本进行采样以减少数据量
            sample_step = max(1, int(1 / args.sample_ratio))  # 根据采样比例确定步长
            
            for i in range(0, num_samples - 85, sample_step):
                try:
                    LLL = product_samples[i:i + 86, 2:6].astype(np.float32)
                    LLLL = product_samples[i:i + 86, 8:9].astype(np.float32)
                    LLLLL = product_samples[i:i + 86, 9:10].astype(np.float32)
                except (ValueError, TypeError):
                    continue
                
                # 数据质量检查
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
                                batch_labels.append(float(lll))
                                batch_samples.append(sample[:, [0, 1, 2, 3, 5, -1]])
                except (ValueError, TypeError, IndexError):
                    continue
        
        # 将批次数据添加到总数据中
        if batch_samples:
            all_samples.extend(batch_samples)
            all_labels.extend(batch_labels)
        
        # 强制垃圾回收
        del batch_samples, batch_labels
        gc.collect()
        
        if (batch_end) % 500 == 0:
            elapsed = time.time() - data_start_time
            eta = elapsed * (total_groups / batch_end - 1)
            logger.info(f"已处理 {batch_end}/{total_groups} 只股票, 当前样本数: {len(all_samples)}, 预计剩余时间: {eta/60:.1f} 分钟")
    
    # 转换为numpy数组并使用float32节省内存
    train_data = np.array(all_samples, dtype=np.float32)
    train_label = np.array(all_labels, dtype=np.float32)
    
    data_process_time = time.time() - data_start_time
    logger.info(f"训练数据处理完成: {train_data.shape}, 耗时: {data_process_time/60:.1f} 分钟")
    logger.info(f"训练样本数: {len(all_samples)}, 标签数: {len(all_labels)}")
    
    # 清理临时变量
    del all_samples, all_labels
    gc.collect()
    
    return train_data, train_label

# 神经网络模型定义
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(60 * 6, 1000)  # 减少第一层神经元数量
        self.fc2 = nn.Linear(1000, 500)    # 减少第二层神经元数量
        self.fc3 = nn.Linear(500, 100)     # 减少第三层神经元数量
        self.fc4 = nn.Linear(100, 1)
        
        # 1D convolution layers
        self.conv1d_1 = nn.Conv1d(6, 16, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        
        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        
        # LSTM layers - 减少层数和隐藏单元
        self.lstm = nn.LSTM(6, 32, 2, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Batch normalization and dropout
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout(0.3)
        
        # Feature combination layer - 动态计算维度
        # fc_out: 1000, conv1d_out: 32, conv2d_out: 32, lstm_out: 64
        self.feature_combine = nn.Linear(1000 + 32 + 32 + 64, 200)
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
        
        # 动态调整feature_combine层的输入维度
        if not hasattr(self, '_feature_dim_adjusted'):
            actual_dim = combined_features.size(1)
            self.feature_combine = nn.Linear(actual_dim, 200).to(combined_features.device)
            self._feature_dim_adjusted = True
        
        combined_out = torch.relu(self.feature_combine(combined_features))
        combined_out = self.dropout(combined_out)
        
        output = self.final_output(combined_out)
        return output

# 内存优化的数据集类
class OptimizedStockDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor([self.labels[idx]])

# 加载数据
df = load_and_sample_data(sample_ratio=args.sample_ratio)
train_data, train_label = process_data_batched(df, batch_size=100)

# 数据标准化
from sklearn.preprocessing import StandardScaler

logger.info("开始数据标准化...")
original_shape = train_data.shape
scaler_features = StandardScaler()
scaler_labels = StandardScaler()

# 重塑数据进行标准化
train_data_reshaped = train_data.reshape(-1, train_data.shape[-1])
train_data_scaled = scaler_features.fit_transform(train_data_reshaped)
train_data_scaled = train_data_scaled.reshape(original_shape)

train_label_scaled = scaler_labels.fit_transform(train_label.reshape(-1, 1)).flatten()

logger.info("数据标准化完成")

# 5折交叉验证
kfold = TimeSeriesSplit(n_splits=3)  # 改为3折以减少内存压力
all_fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data_scaled)):
    logger.info(f"\n开始第 {fold+1}/3 折训练")
    
    # 创建数据集
    X_train_fold = train_data_scaled[train_idx]
    y_train_fold = train_label_scaled[train_idx]
    X_val_fold = train_data_scaled[val_idx]
    y_val_fold = train_label_scaled[val_idx]
    
    # 创建数据加载器，使用更小的批次大小
    train_dataset = OptimizedStockDataset(X_train_fold, y_train_fold)
    val_dataset = OptimizedStockDataset(X_val_fold, y_val_fold)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 初始化模型
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
    
    # 训练模型
    best_corr = -float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    aucs = []
    
    num_epochs = 30  # 减少训练轮数
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_data, batch_labels in train_progress:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
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
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
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
        
        # 强制垃圾回收
        if epoch % 5 == 0:
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
    gc.collect()

# 打印所有折的结果
logger.info("\n=== 3折交叉验证结果汇总 ===")
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
    'sample_ratio': args.sample_ratio,
    'batch_size': args.batch_size
}

with open('./weights/cv_results_optimized.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

logger.info("结果已保存到 ./weights/cv_results_optimized.json")