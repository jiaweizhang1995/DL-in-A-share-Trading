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
import psutil
import h5py

if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('5_fold_cv_memory_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 命令行参数解析
parser = argparse.ArgumentParser(description='5-Fold Cross-Validation with Memory Optimization (Full Precision)')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from')
parser.add_argument('--checkpoint-freq', type=int, default=5, help='Save checkpoint every N epochs')
parser.add_argument('--resume-fold', type=int, default=None, help='Resume from specific fold (0-based)')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training')
parser.add_argument('--chunk-size', type=int, default=500, help='Number of stocks to process at once')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")
logger.info(f"开始5折交叉验证 (内存优化版本 - 保持完整精度) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"批次大小: {args.batch_size}, 数据块大小: {args.chunk_size}")

def log_memory_usage(stage=""):
    """记录内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"内存使用 {stage}: {memory_mb:.1f} MB")

# 与原版完全相同的神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # Fully connected layers - 保持原版配置
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
        
        # LSTM layers - 保持原版配置
        self.lstm = nn.LSTM(6, 64, 4, batch_first=True, bidirectional=True, dropout=0.2)
        
        # Batch normalization and dropout
        self.bn1 = nn.BatchNorm1d(2000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout(0.3)
        
        # Feature combination layer - 需要在第一次前向传播时动态确定维度
        self.feature_combine = None
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
        
        # 动态创建feature_combine层
        if self.feature_combine is None:
            feature_dim = combined_features.size(1)
            self.feature_combine = nn.Linear(feature_dim, 200).to(combined_features.device)
            logger.info(f"动态创建特征组合层，输入维度: {feature_dim}")
        
        combined_out = torch.relu(self.feature_combine(combined_features))
        combined_out = self.dropout(combined_out)
        
        output = self.final_output(combined_out)
        return output

# 内存优化的数据处理类
class MemoryOptimizedDataProcessor:
    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size
        self.temp_file = './temp_processed_data.h5'
        
    def load_and_filter_data(self):
        """加载和过滤数据，与原版逻辑完全一致"""
        logger.info("开始加载和预处理数据...")
        log_memory_usage("开始加载数据前")
        
        df = pd.read_csv('./data/20250701.csv')
        logger.info(f"原始数据加载完成: {df.shape[0]} 条记录")
        log_memory_usage("原始数据加载后")

        df['amount'] = df['amount'] / 10000000
        original_stocks = df['stock_id'].nunique()
        
        # 与原版完全相同的过滤逻辑
        df = df.groupby('stock_id').filter(lambda x: len(x) >= 180)
        df = df.groupby('stock_id').apply(lambda x: x.iloc[20:], include_groups=False)
        df.reset_index(inplace=True)
        df.drop('level_1', axis=1, inplace=True)
        logger.info(f"过滤后保留 {df['stock_id'].nunique()}/{original_stocks} 只股票")

        df['stock_id'] = df['stock_id'].astype(str)
        df = df[~df['stock_id'].str.startswith('8')]
        df = df[~df['stock_id'].str.startswith('68')]
        df = df[~df['stock_id'].str.startswith('4')]
        df['date'] = pd.to_datetime(df['date'])
        
        # 与原版相同的时间范围
        df = df[(df['date'] > pd.to_datetime('2010-01-01')) & (df['date'] < pd.to_datetime('2024-01-01'))]
        logger.info(f"最终处理后: {df['stock_id'].nunique()} 只股票, {df.shape[0]} 条记录")
        log_memory_usage("数据过滤后")
        
        return df
    
    def process_data_in_chunks(self, df):
        """分块处理数据，避免内存溢出"""
        logger.info("开始分块处理训练数据样本...")
        data_start_time = time.time()
        
        grouped = df.groupby('stock_id')
        total_groups = len(grouped)
        logger.info(f"需要处理 {total_groups} 只股票的数据")
        
        # 删除旧的临时文件
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        
        all_samples_count = 0
        groups_list = list(grouped)
        
        # 分块处理
        for chunk_start in range(0, total_groups, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_groups)
            chunk_samples = []
            chunk_labels = []
            
            logger.info(f"处理第 {chunk_start//self.chunk_size + 1} 块 ({chunk_start}-{chunk_end})...")
            
            for idx in range(chunk_start, chunk_end):
                stock_id, group = groups_list[idx]
                product_samples = group.values
                num_samples = len(product_samples)
                
                if num_samples < 180:
                    continue
                
                # 与原版完全相同的样本生成逻辑
                for i in range(num_samples - 85):
                    try:
                        LLL = product_samples[i:i + 86, 2:6].astype(np.float32)
                        LLLL = product_samples[i:i + 86, 8:9].astype(np.float32)
                        LLLLL = product_samples[i:i + 86, 9:10].astype(np.float32)
                    except (ValueError, TypeError):
                        continue
                    
                    # 与原版相同的数据质量检查
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
                                    chunk_labels.append(float(lll))
                                    chunk_samples.append(sample[:, [0, 1, 2, 3, 5, -1]])
                    except (ValueError, TypeError, IndexError):
                        continue
            
            # 保存当前块到临时文件
            if chunk_samples:
                chunk_data = np.array(chunk_samples, dtype=np.float32)
                chunk_labels_array = np.array(chunk_labels, dtype=np.float32)
                
                with h5py.File(self.temp_file, 'a') as f:
                    if 'samples' not in f:
                        # 创建可扩展的数据集
                        f.create_dataset('samples', data=chunk_data, maxshape=(None, 60, 6), 
                                       compression='gzip', compression_opts=9)
                        f.create_dataset('labels', data=chunk_labels_array, maxshape=(None,),
                                       compression='gzip', compression_opts=9)
                    else:
                        # 扩展现有数据集
                        current_size = f['samples'].shape[0]
                        new_size = current_size + len(chunk_data)
                        f['samples'].resize((new_size, 60, 6))
                        f['labels'].resize((new_size,))
                        f['samples'][current_size:] = chunk_data
                        f['labels'][current_size:] = chunk_labels_array
                
                all_samples_count += len(chunk_samples)
                logger.info(f"块 {chunk_start//self.chunk_size + 1} 处理完成，样本数: {len(chunk_samples)}, 总样本数: {all_samples_count}")
            
            # 清理当前块的内存
            del chunk_samples, chunk_labels
            if 'chunk_data' in locals():
                del chunk_data, chunk_labels_array
            gc.collect()
            log_memory_usage(f"处理块 {chunk_start//self.chunk_size + 1} 后")
            
            if (chunk_end) % 1000 == 0:
                elapsed = time.time() - data_start_time
                eta = elapsed * (total_groups / chunk_end - 1)
                logger.info(f"已处理 {chunk_end}/{total_groups} 只股票, 当前总样本数: {all_samples_count}, 预计剩余时间: {eta/60:.1f} 分钟")
        
        data_process_time = time.time() - data_start_time
        logger.info(f"训练数据处理完成: 总样本数 {all_samples_count}, 耗时: {data_process_time/60:.1f} 分钟")
        log_memory_usage("数据处理完成后")
        
        return all_samples_count

# 内存优化的数据集类
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, indices=None):
        self.hdf5_file = hdf5_file
        self.indices = indices
        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(indices) if indices is not None else f['samples'].shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx] if self.indices is not None else idx
        with h5py.File(self.hdf5_file, 'r') as f:
            sample = f['samples'][actual_idx]
            label = f['labels'][actual_idx]
        return torch.FloatTensor(sample), torch.FloatTensor([label])

# 主处理流程
log_memory_usage("程序开始")

# 创建数据处理器
processor = MemoryOptimizedDataProcessor(chunk_size=args.chunk_size)

# 加载和处理数据
df = processor.load_and_filter_data()
total_samples = processor.process_data_in_chunks(df)

# 清理DataFrame内存
del df
gc.collect()
log_memory_usage("DataFrame清理后")

# 数据标准化
logger.info("开始数据标准化...")
from sklearn.preprocessing import StandardScaler

scaler_features = StandardScaler()
scaler_labels = StandardScaler()

# 分批进行标准化以节省内存
batch_size_for_scaling = 10000
scaling_start_time = time.time()

with h5py.File(processor.temp_file, 'r') as f:
    # 计算统计信息
    n_samples = f['samples'].shape[0]
    n_batches = (n_samples + batch_size_for_scaling - 1) // batch_size_for_scaling
    
    logger.info(f"准备标准化 {n_samples} 个样本，分 {n_batches} 批处理，每批 {batch_size_for_scaling} 个样本")
    
    # 第一遍：计算均值和方差
    logger.info("第一遍：计算数据统计信息...")
    for i in range(n_batches):
        start_idx = i * batch_size_for_scaling
        end_idx = min((i + 1) * batch_size_for_scaling, n_samples)
        
        batch_data = f['samples'][start_idx:end_idx]
        batch_labels = f['labels'][start_idx:end_idx]
        
        # 重塑数据进行标准化
        batch_data_reshaped = batch_data.reshape(-1, batch_data.shape[-1])
        
        if i == 0:
            scaler_features.partial_fit(batch_data_reshaped)
            scaler_labels.partial_fit(batch_labels.reshape(-1, 1))
        else:
            scaler_features.partial_fit(batch_data_reshaped)
            scaler_labels.partial_fit(batch_labels.reshape(-1, 1))
        
        # 定期报告进度
        if (i + 1) % 100 == 0 or i == n_batches - 1:
            elapsed = time.time() - scaling_start_time
            progress = (i + 1) / n_batches
            eta = elapsed / progress - elapsed if progress > 0 else 0
            logger.info(f"统计信息计算进度: {i+1}/{n_batches} ({progress*100:.1f}%), "
                       f"已耗时: {elapsed/60:.1f}分钟, 预计剩余: {eta/60:.1f}分钟")
            log_memory_usage(f"统计计算批次 {i+1}")

logger.info("统计信息计算完成，开始应用标准化...")

# 创建标准化后的数据文件
scaled_file = './temp_scaled_data.h5'
if os.path.exists(scaled_file):
    os.remove(scaled_file)

logger.info("第二遍：应用数据标准化...")
transform_start_time = time.time()

with h5py.File(processor.temp_file, 'r') as f_in, h5py.File(scaled_file, 'w') as f_out:
    # 创建标准化后的数据集
    logger.info("创建标准化后的数据文件...")
    f_out.create_dataset('samples', (n_samples, 60, 6), dtype=np.float32,
                        compression='gzip', compression_opts=9)
    f_out.create_dataset('labels', (n_samples,), dtype=np.float32,
                        compression='gzip', compression_opts=9)
    logger.info("数据文件创建完成，开始标准化转换...")
    
    for i in range(n_batches):
        start_idx = i * batch_size_for_scaling
        end_idx = min((i + 1) * batch_size_for_scaling, n_samples)
        
        # 读取数据
        batch_data = f_in['samples'][start_idx:end_idx]
        batch_labels = f_in['labels'][start_idx:end_idx]
        
        # 标准化数据
        original_shape = batch_data.shape
        batch_data_reshaped = batch_data.reshape(-1, batch_data.shape[-1])
        batch_data_scaled = scaler_features.transform(batch_data_reshaped)
        batch_data_scaled = batch_data_scaled.reshape(original_shape)
        
        batch_labels_scaled = scaler_labels.transform(batch_labels.reshape(-1, 1)).flatten()
        
        # 保存标准化后的数据
        f_out['samples'][start_idx:end_idx] = batch_data_scaled
        f_out['labels'][start_idx:end_idx] = batch_labels_scaled
        
        # 定期报告进度
        if (i + 1) % 50 == 0 or i == n_batches - 1:
            elapsed = time.time() - transform_start_time
            progress = (i + 1) / n_batches
            eta = elapsed / progress - elapsed if progress > 0 else 0
            logger.info(f"标准化转换进度: {i+1}/{n_batches} ({progress*100:.1f}%), "
                       f"已耗时: {elapsed/60:.1f}分钟, 预计剩余: {eta/60:.1f}分钟")
            log_memory_usage(f"标准化批次 {i+1}")
            
            # 显示写入的样本范围
            logger.info(f"已处理样本范围: {start_idx} - {end_idx}, 当前批次大小: {end_idx - start_idx}")

logger.info("数据标准化完成")
log_memory_usage("标准化完成后")

# 删除临时文件
os.remove(processor.temp_file)

# 5折交叉验证
kfold = TimeSeriesSplit(n_splits=5)
all_fold_results = []

# 创建索引数组
indices = np.arange(total_samples)

for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
    logger.info(f"\n开始第 {fold+1}/5 折训练")
    log_memory_usage(f"第 {fold+1} 折开始前")
    
    # 创建数据集
    train_dataset = HDF5Dataset(scaled_file, train_idx)
    val_dataset = HDF5Dataset(scaled_file, val_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=0, pin_memory=False)
    
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
    
    num_epochs = 50  # 与原版相同的训练轮数
    
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
        
        # 定期垃圾回收
        if epoch % 10 == 0:
            gc.collect()
            log_memory_usage(f"第 {fold+1} 折 Epoch {epoch+1}")
    
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
    gc.collect()
    log_memory_usage(f"第 {fold+1} 折清理后")

# 清理临时文件
os.remove(scaled_file)

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
    'total_samples': total_samples,
    'batch_size': args.batch_size,
    'chunk_size': args.chunk_size
}

with open('./weights/cv_results_memory_optimized.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

logger.info("结果已保存到 ./weights/cv_results_memory_optimized.json")
log_memory_usage("程序结束")