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
if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('5_fold_cv.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 命令行参数解析
parser = argparse.ArgumentParser(description='5-Fold Cross-Validation with Checkpoint Support')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from')
parser.add_argument('--checkpoint-freq', type=int, default=5, help='Save checkpoint every N epochs')
parser.add_argument('--resume-fold', type=int, default=None, help='Resume from specific fold (0-based)')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")
logger.info(f"开始5折交叉验证 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if args.resume:
    logger.info(f"将从 checkpoint 恢复训练: {args.resume}")
else:
    logger.info("从头开始训练")

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
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
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

# 重塑为 2D 进行标准化
train_data_2d = train_data.reshape(-1, train_data.shape[-1])
train_data_scaled = scaler_features.fit_transform(train_data_2d)
train_data_scaled = train_data_scaled.reshape(original_shape)

train_label_scaled = scaler_labels.fit_transform(train_label.reshape(-1, 1)).flatten()

logger.info("数据标准化完成")
logger.info(f"训练数据范围: [{train_data_scaled.min():.3f}, {train_data_scaled.max():.3f}]")
logger.info(f"训练标签范围: [{train_label_scaled.min():.3f}, {train_label_scaled.max():.3f}]")

end_time = time.time()
execution_time = end_time - data_start_time
logger.info(f"总数据处理时间: {execution_time/60:.1f} 分钟")

MODEL = r'APP'
batch_size = 512  # 使用和Training.py相同的batch size
learning_rate = 0.001  # 使用和Training.py相同的learning rate
N = train_data_scaled.shape[0]
num_epochs = 200  # 减少epochs以加快测试
k = 3
tscv = TimeSeriesSplit(n_splits=k)
softmax_function = nn.Softmax(dim=1)

logger.info(f"开始{k}折交叉验证训练")
logger.info(f"训练参数: batch_size={batch_size}, learning_rate={learning_rate}, num_epochs={num_epochs}")
logger.info(f"数据集大小: {N}")

L5 = []
L5_v = []
T_AUC = []

# 初始化恢复状态
resume_fold = 0
all_fold_results = []

# 如果恢复训练，加载 checkpoint
if args.resume:
    checkpoint_result = load_checkpoint(args.resume)
    if checkpoint_result is not None:
        resume_fold, _, _, _, _, _, _, all_fold_results, checkpoint_dict = checkpoint_result
        L5 = all_fold_results.get('L5', [])
        L5_v = all_fold_results.get('L5_v', [])
        T_AUC = all_fold_results.get('T_AUC', [])
        logger.info(f"恢复训练从 fold {resume_fold+1} 开始")

cv_start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(tscv.split(np.arange(N))):
    if fold < resume_fold:
        continue
    
    fold_start_time = time.time()
    logger.info(f"\n=== 开始 Fold {fold + 1}/{k} ===")
    logger.info(f"训练集: {len(train_idx)} 样本, 验证集: {len(test_idx)} 样本")
    
    Train_data = train_data_scaled[train_idx]
    Train_label = train_label_scaled[train_idx]

    Test_data = train_data_scaled[test_idx]
    Test_label = train_label_scaled[test_idx]

    Train_data = torch.tensor(Train_data, dtype=torch.float32)
    Train_label = torch.tensor(Train_label, dtype=torch.float32)

    Test_data = torch.tensor(Test_data, dtype=torch.float32)
    Test_label = torch.tensor(Test_label, dtype=torch.float32)

    dataset_train = TensorDataset(Train_data, Train_label)
    dataset_val = TensorDataset(Test_data, Test_label)
    train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)

    class model(nn.Module):
        def __init__(self,
                     fc1_size=2000,
                     fc2_size=1000,
                     fc3_size=100,
                     fc1_dropout=0.2,
                     fc2_dropout=0.2,
                     fc3_dropout=0.2,
                     num_of_classes=1):
            super(model, self).__init__()

            self.f_model = nn.Sequential(
                nn.Linear(3296, fc1_size),  # 887
                nn.BatchNorm1d(fc1_size),
                nn.ReLU(),
                nn.Dropout(fc1_dropout),
                nn.Linear(fc1_size, fc2_size),
                nn.BatchNorm1d(fc2_size),
                nn.ReLU(),
                nn.Dropout(fc2_dropout),
                nn.Linear(fc2_size, fc3_size),
                nn.BatchNorm1d(fc3_size),
                nn.ReLU(),
                nn.Dropout(fc3_dropout),
                nn.Linear(fc3_size, num_of_classes),

            )

            self.conv_layers1 = nn.Sequential(
                nn.Conv1d(6, 16, kernel_size=1),
                nn.BatchNorm1d(16),
                nn.Dropout(fc3_dropout),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(16, 32, kernel_size=1),
                nn.BatchNorm1d(32),
                nn.Dropout(fc3_dropout),
                nn.ReLU(),
            )

            self.conv_2D = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=2),
                nn.BatchNorm2d(16),
                nn.Dropout(fc3_dropout),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=2),
                nn.BatchNorm2d(32),
                nn.Dropout(fc3_dropout),
                nn.ReLU(),
            )
            hidden_dim = 32
            self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=4, batch_first=True,
                                # dropout=fc3_dropout,
                                bidirectional=True)
            hidden_dim = 1
            self.l = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=4, batch_first=True,
                             # dropout=fc3_dropout,
                             bidirectional=True)
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

        def forward(self, x):
            apply = torch.narrow(x, dim=-1, start=0, length=1)[:, -90:, ].squeeze(1)
            redeem = torch.narrow(x, dim=-1, start=1, length=1)[:, -90:, ].squeeze(1)
            apply, _ = self.l(apply)
            redeem, _ = self.l(redeem)
            apply = torch.reshape(apply, (apply.shape[0], apply.shape[1] * apply.shape[2]))
            redeem = torch.reshape(redeem, (redeem.shape[0], redeem.shape[1] * redeem.shape[2]))

            ZFF = torch.narrow(x, dim=-1, start=2, length=1)[:, -90:, ].squeeze(1)
            HS = torch.narrow(x, dim=-1, start=3, length=1)[:, -90:, ].squeeze(1)
            ZFF, _ = self.l(ZFF)
            HS, _ = self.l(HS)
            ZFF = torch.reshape(ZFF, (ZFF.shape[0], ZFF.shape[1] * ZFF.shape[2]))
            HS = torch.reshape(HS, (HS.shape[0], HS.shape[1] * HS.shape[2]))

            min_vals, _ = torch.min(x, dim=1, keepdim=True)
            max_vals, _ = torch.max(x, dim=1, keepdim=True)
            x = (x - min_vals) / (max_vals - min_vals + 0.00001)

            xx = x.unsqueeze(1)
            xx = self.conv_2D(xx)
            xx = torch.reshape(xx, (xx.shape[0], xx.shape[1] * xx.shape[2] * xx.shape[3]))
            x = x.transpose(1, 2)
            x = self.conv_layers1(x)
            out = x.transpose(1, 2)
            out2, _ = self.lstm(out)
            out2 = torch.reshape(out2, (out2.shape[0], out2.shape[1] * out2.shape[2]))

            IN = torch.cat((xx, out2, apply, redeem, ZFF, HS), dim=1)
            out = self.f_model(IN)
            return out


    model = model()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    criterion = nn.SmoothL1Loss()  # 使用和Training.py相同的loss函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # 使用和Training.py相同的optimizer
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    # 初始化训练变量
    start_epoch = 0
    L_train = []
    L_val = []
    AUC = []
    min_validation_loss = 0
    best_epoch = -1
    
    # 如果恢复训练，加载特定fold的checkpoint
    if args.resume and fold == resume_fold:
        checkpoint_result = load_checkpoint(args.resume)
        if checkpoint_result is not None:
            _, start_epoch, L_train, L_val, AUC, min_validation_loss, best_epoch, _, checkpoint_dict = checkpoint_result
            model.load_state_dict(checkpoint_dict['model_state_dict'])
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            logger.info(f"恢复 Fold {fold+1} 从 epoch {start_epoch+1} 开始")
    
    logger.info(f"开始训练 Fold {fold+1}: {num_epochs} 个 epoch")
    training_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        logger.info(f"\n=== Fold {fold+1} Epoch {epoch+1}/{num_epochs} ===")
        
        train_running_loss = 0.0
        counter = 0
        model.train()
        for seq, y in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Training"):
            counter += 1
            output = model(seq.to(device))
            loss = criterion(output.squeeze(), y.to(device))
            
            # 检查损失值是否异常
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"检测到异常损失值: {loss.item()}, 跳过此批次")
                continue
                
            train_running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        TL = train_running_loss / counter
        L_train.append(TL)
        
        logger.info(f"训练损失: {TL:.6f}")
        
        model.eval()
        PREDICT = []
        TRUE = []
        counter = 0
        with torch.no_grad():
            current_test_loss = 0.0
            for SEQ, Z in tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Validation"):
                counter += 1
                output = model(SEQ.to(device))
                loss = criterion(output.squeeze(), Z.to(device))
                
                # 检查损失值是否异常
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"验证检测到异常损失值: {loss.item()}, 跳过此批次")
                    continue
                    
                current_test_loss += loss.item()
                PREDICT.extend(output.cpu().numpy())
                TRUE.extend(Z.cpu().numpy())
            T_loss = current_test_loss / counter
            L_val.append(T_loss)
            PP = np.array(PREDICT)
            TT = np.array(TRUE)
            flattened_array1 = PP.flatten()
            flattened_array2 = TT.flatten()
            
            # 检查是否有有效预测结果
            if len(flattened_array1) == 0 or len(flattened_array2) == 0:
                logger.warning("没有有效的预测结果")
                corr = 0.0
            else:
                try:
                    correlation_matrix = np.corrcoef(flattened_array1, flattened_array2)
                    corr = correlation_matrix[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                except:
                    logger.warning("相关系数计算失败")
                    corr = 0.0
            
            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - training_start_time
            avg_epoch_time = elapsed_time / (epoch - start_epoch + 1)
            eta_fold = avg_epoch_time * (num_epochs - epoch - 1)
            
            # 计算整个CV的ETA
            elapsed_cv_time = time.time() - cv_start_time
            total_epochs_done = fold * num_epochs + (epoch - start_epoch + 1)
            total_epochs = k * num_epochs
            avg_epoch_time_cv = elapsed_cv_time / total_epochs_done
            eta_cv = avg_epoch_time_cv * (total_epochs - total_epochs_done)
            
            logger.info(f"验证损失: {T_loss:.6f}, 相关系数: {corr:.6f}")
            logger.info(f"Epoch 耗时: {epoch_time/60:.1f} 分钟, Fold ETA: {eta_fold/60:.1f} 分钟, CV ETA: {eta_cv/60:.1f} 分钟")

            ################################################################################################################
            if min_validation_loss < corr:
                min_validation_loss = corr
                best_epoch = epoch
                logger.info(f"★ 新的最佳相关系数: {min_validation_loss:.6f} (Fold {fold+1} Epoch {best_epoch+1})")
                torch.save(model.state_dict(), fr"./weights/model_{MODEL}_{fold}.pt")
                logger.info(f"模型已保存到: ./weights/model_{MODEL}_{fold}.pt")
            AUC.append(corr)
            
            # 每 N 个 epoch 保存 checkpoint
            if (epoch + 1) % args.checkpoint_freq == 0:
                checkpoint_path = f'./checkpoints/cv_fold_{fold}_epoch_{epoch+1}.pth'
                all_fold_results_current = {'L5': L5, 'L5_v': L5_v, 'T_AUC': T_AUC}
                save_checkpoint(fold, epoch, model, optimizer, scheduler, L_train, L_val, AUC, 
                              min_validation_loss, best_epoch, all_fold_results_current, checkpoint_path)
            
            # 最新 checkpoint (总是保存最新的)
            latest_checkpoint_path = f'./checkpoints/cv_fold_{fold}_latest.pth'
            all_fold_results_current = {'L5': L5, 'L5_v': L5_v, 'T_AUC': T_AUC}
            save_checkpoint(fold, epoch, model, optimizer, scheduler, L_train, L_val, AUC, 
                           min_validation_loss, best_epoch, all_fold_results_current, latest_checkpoint_path)

    # Fold 完成
    fold_time = time.time() - fold_start_time
    logger.info(f"Fold {fold+1} 完成, 耗时: {fold_time/60:.1f} 分钟")
    logger.info(f"Fold {fold+1} 最佳相关系数: {min_validation_loss:.6f} (Epoch {best_epoch+1})")
    
    L5.append(L_train)
    L5_v.append(L_val)
    T_AUC.append(AUC)

# 交叉验证完成
total_cv_time = time.time() - cv_start_time
logger.info(f"\n=== {k}折交叉验证完成 ===")
logger.info(f"总训练时间: {total_cv_time/3600:.1f} 小时")

# 计算平均性能
avg_best_corr = np.mean([max(fold_aucs) for fold_aucs in T_AUC])
std_best_corr = np.std([max(fold_aucs) for fold_aucs in T_AUC])
logger.info(f"平均最佳相关系数: {avg_best_corr:.6f} ± {std_best_corr:.6f}")

# 每个fold的最佳结果
for i, fold_aucs in enumerate(T_AUC):
    best_corr_fold = max(fold_aucs)
    best_epoch_fold = fold_aucs.index(best_corr_fold)
    logger.info(f"Fold {i+1} 最佳相关系数: {best_corr_fold:.6f} (Epoch {best_epoch_fold+1})")

logger.info("开始生成交叉验证结果图表...")
fig1, ax1 = plt.subplots()
for i, fold_train_loss in enumerate(L5):
    ax1.plot(fold_train_loss, label=f"Fold {i + 1}")
ax1.legend(loc='upper right')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title(f'{k}-fold cross-validation training losses - Avg Best Corr: {avg_best_corr:.4f}')

# 绘制验证损失函数折线图
fig2, ax2 = plt.subplots()
for i, fold_val_loss in enumerate(L5_v):
    ax2.plot(fold_val_loss, label=f"Fold {i + 1}")
ax2.legend(loc='upper right')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title(f'{k}-fold cross-validation validation losses - Time: {total_cv_time/3600:.1f}h')

fig3, ax3 = plt.subplots()
for i, fold_val_loss in enumerate(T_AUC):
    best_corr_fold = max(fold_val_loss)
    ax3.plot(fold_val_loss, label=f"Fold {i + 1} (Max: {best_corr_fold:.4f})")
ax3.legend(loc='upper right')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Test-set corr')
ax3.set_title(f'Test-set correlation - Avg: {avg_best_corr:.4f} ± {std_best_corr:.4f}')
ax3.axhline(y=avg_best_corr, color='r', linestyle='--', alpha=0.7, label=f'Avg: {avg_best_corr:.4f}')

fig1.savefig(f"./{MODEL}_CV_Training_Loss.png")
fig2.savefig(f"./{MODEL}_CV_Validation_Loss.png")
fig3.savefig(f"./{MODEL}_CV_Correlation.png")

logger.info(f"结果图表已保存: {MODEL}_CV_*.png")
logger.info(f"训练日志已保存: 5_fold_cv.log")
logger.info(f"{k}折交叉验证完成! - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
