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
if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 命令行参数解析
parser = argparse.ArgumentParser(description='Training Script with Checkpoint Support')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from')
parser.add_argument('--checkpoint-freq', type=int, default=5, help='Save checkpoint every N epochs')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")
logger.info(f"开始训练流程 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
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
    train_label_scaled = data['train_label']
    test_data_scaled = data['test_data']
    test_label_scaled = data['test_label']
    logger.info(f"加载完成 - 训练: {train_data_scaled.shape}, 测试: {test_data_scaled.shape}")
else:
    logger.info("开始加载和预处理数据...")
df = pd.read_csv('./data/20250701.csv')
logger.info(f"原始数据加载完成: {df.shape[0]} 条记录")

df['amount'] = df['amount'] / 10000000
original_stocks = df['stock_id'].nunique()
df = df.groupby('stock_id').filter(lambda x: len(x) >= 180)  # 少于180交易日的股票不要
df = df.groupby('stock_id').apply(lambda x: x.iloc[20:])  # 刚上市的20个交易日不要
df.reset_index(drop=True, inplace=True)
logger.info(f"过滤后保留 {df['stock_id'].nunique()}/{original_stocks} 只股票")

df['stock_id'] = df['stock_id'].astype(str)
df = df[~df['stock_id'].str.startswith('8')]
df = df[~df['stock_id'].str.startswith('68')]
df = df[~df['stock_id'].str.startswith('4')]
df['date'] = pd.to_datetime(df['date'])
logger.info(f"最终处理后: {df['stock_id'].nunique()} 只股票, {df.shape[0]} 条记录")

df_train = df[(df['date'] > pd.to_datetime('2010-01-01')) & (df['date'] < pd.to_datetime('2024-01-01'))]
df_test = df[df['date'] >= pd.to_datetime('2024-01-01')]
logger.info(f"训练集: {df_train.shape[0]} 条记录 ({df_train['stock_id'].nunique()} 只股票)")
logger.info(f"测试集: {df_test.shape[0]} 条记录 ({df_test['stock_id'].nunique()} 只股票)")

logger.info("开始处理训练数据样本...")
data_start_time = time.time()
grouped = df_train.groupby('stock_id')
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
        LLL = product_samples[i:i + 86, 2:6].astype(np.float32)
        LLLL = product_samples[i:i + 86, 11:12].astype(np.float32)
        LLLLL = product_samples[i:i + 86, 9:10].astype(np.float32)
        if (np.any(np.isnan(LLL)) or np.any(LLL <= 0) or np.any(np.isnan(LLLL)) or
                np.any(LLLL < 0.1) or np.any(LLLL > 40) or
                np.any(LLLLL > 20) or np.any(LLLLL < -10)):
            pass
        else:
            sample = product_samples[i:i + 60, 2:-1]

            l = product_samples[i + 60:i + 61, 2:3]
            Hl = product_samples[i + 60:i + 61, 4:5]
            if l != Hl:
                ll = np.mean(product_samples[i + 61:i + 86, 5:6])
                lll = ((ll - l) / l)*100
                label.append(lll)
                samples.append(sample[ :, [0, 1, 2, 3, 4, -1]]) #开、收、高、低、成交额、换手率

train_data = np.array(samples)
train_data = train_data.astype(np.float32)
train_label = np.array(label).astype(np.float32)
data_process_time = time.time() - data_start_time
logger.info(f"训练数据处理完成: {train_data.shape}, 耗时: {data_process_time/60:.1f} 分钟")
logger.info(f"训练样本数: {len(samples)}, 标签数: {len(label)}")
logger.info("开始处理测试数据样本...")
test_start_time = time.time()
grouped_test = df_test.groupby('stock_id')
samples = []
label = []
total_test_groups = len(grouped_test)
logger.info(f"需要处理 {total_test_groups} 只股票的测试数据")

for idx, (stock_id, group) in enumerate(tqdm(grouped_test, desc="处理测试数据")):
    product_samples = group.values
    num_samples = len(product_samples)
    if num_samples < 180:
        continue
    for i in range(num_samples - 85):
        LLL = product_samples[i:i + 86, 2:6].astype(np.float32)
        LLLL = product_samples[i:i + 86, 11:12].astype(np.float32)
        LLLLL = product_samples[i:i + 86, 9:10].astype(np.float32)
        if (np.any(np.isnan(LLL)) or np.any(LLL <= 0) or np.any(np.isnan(LLLL)) or
                np.any(LLLL < 0.1) or np.any(LLLL > 40) or
                np.any(LLLLL > 20) or np.any(LLLLL < -10)):
            pass
        else:
            sample = product_samples[i:i + 60, 2:-1]

            l = product_samples[i + 60:i + 61, 2:3]
            Hl = product_samples[i + 60:i + 61, 4:5]
            if l != Hl:
                ll = np.mean(product_samples[i + 61:i + 86, 5:6])
                lll = ((ll - l) / l)*100
                label.append(lll)
                samples.append(sample[ :, [0, 1, 2, 3, 4, -1]]) #开、收、高、低、成交额、换手率

test_data = np.array(samples)
test_data = test_data.astype(np.float32)
test_label = np.array(label).astype(np.float32)
test_process_time = time.time() - test_start_time
total_data_time = time.time() - data_start_time
logger.info(f"测试数据处理完成: {test_data.shape}, 耗时: {test_process_time/60:.1f} 分钟")
logger.info(f"测试样本数: {len(samples)}, 标签数: {len(label)}")
logger.info(f"总数据处理时间: {total_data_time/60:.1f} 分钟")

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

# 标准化测试数据
test_original_shape = test_data.shape
test_data_2d = test_data.reshape(-1, test_data.shape[-1])
test_data_scaled = scaler_features.transform(test_data_2d)
test_data_scaled = test_data_scaled.reshape(test_original_shape)

test_label_scaled = scaler_labels.transform(test_label.reshape(-1, 1)).flatten()

logger.info("数据标准化完成")
logger.info(f"训练数据范围: [{train_data_scaled.min():.3f}, {train_data_scaled.max():.3f}]")
logger.info(f"训练标签范围: [{train_label_scaled.min():.3f}, {train_label_scaled.max():.3f}]")

# 保存预处理数据以供下次使用
logger.info("保存预处理数据...")
np.savez('./processed_data.npz',
         train_data=train_data_scaled,
         train_label=train_label_scaled,
         test_data=test_data_scaled,
         test_label=test_label_scaled)
logger.info("预处理数据已保存到 processed_data.npz")

MODEL = r'baseline'
batch_size = 512  # 降低批大小避免内存问题
learning_rate = 0.00001  # 降低学习率避免梯度爆炸
N = train_data.shape[0]
num_epochs = 50  # 增加 epoch 数量
softmax_function = nn.Softmax(dim=1)



Train_data = torch.tensor(train_data_scaled, dtype=torch.float32)
Train_label = torch.tensor(train_label_scaled, dtype=torch.float32)

Test_data = torch.tensor(test_data_scaled, dtype=torch.float32)
Test_label = torch.tensor(test_label_scaled, dtype=torch.float32)

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
            nn.Linear(fc3_size, 1),

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
                            bidirectional=True)
        hidden_dim = 1
        self.l = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=4, batch_first=True,
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

logger.info("初始化模型和优化器...")
model = model()
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"模型参数总数: {total_params:,}, 可训练参数: {trainable_params:,}")

# 使用 Huber Loss 更稳定
criterion = nn.SmoothL1Loss()  # Huber Loss
# criterion = nn.MSELoss()  # 备用 MSE
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# 初始化训练变量
start_epoch = 0
L_train = []
L_val = []
AUC = []
min_validation_loss = 0
best_epoch = -1

# 如果恢复训练，加载 checkpoint
if args.resume:
    checkpoint_result = load_checkpoint(args.resume, model, optimizer, scheduler)
    if checkpoint_result is not None:
        start_epoch, L_train, L_val, AUC, min_validation_loss, best_epoch = checkpoint_result

logger.info(f"开始训练: {num_epochs} 个 epoch, 批大小: {batch_size}, 学习率: {learning_rate}")
logger.info(f"从 epoch {start_epoch+1} 开始训练")
training_start_time = time.time()

for epoch in range(start_epoch, num_epochs):
    epoch_start_time = time.time()
    logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
    train_running_loss = 0.0
    counter = 0
    model.train()
    
    for seq, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
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
        
        # 梯度裁剪防止梯度爆炸
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
        for SEQ, Z in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
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
        avg_epoch_time = elapsed_time / (epoch + 1)
        eta = avg_epoch_time * (num_epochs - epoch - 1)
        
        logger.info(f"验证损失: {T_loss:.6f}, 相关系数: {corr:.6f}")
        logger.info(f"Epoch 耗时: {epoch_time/60:.1f} 分钟, 已耗时: {elapsed_time/60:.1f} 分钟, ETA: {eta/60:.1f} 分钟")
        
        ################################################################################################################
        if min_validation_loss < corr:
            min_validation_loss = corr
            best_epoch = epoch
            logger.info(f"★ 新的最佳相关系数: {min_validation_loss:.6f} (Epoch {best_epoch+1})")
            torch.save(model.state_dict(), fr"./weights/model_{MODEL}.pt")
            logger.info(f"模型已保存到: ./weights/model_{MODEL}.pt")
        AUC.append(corr)
        
        # 每 N 个 epoch 保存 checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = f'./checkpoints/checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(epoch, model, optimizer, scheduler, L_train, L_val, AUC, 
                          min_validation_loss, best_epoch, checkpoint_path)
        
        # 最新 checkpoint (总是保存最新的)
        latest_checkpoint_path = './checkpoints/checkpoint_latest.pth'
        save_checkpoint(epoch, model, optimizer, scheduler, L_train, L_val, AUC, 
                       min_validation_loss, best_epoch, latest_checkpoint_path)

total_training_time = time.time() - training_start_time
logger.info(f"\n=== 训练完成 ===")
logger.info(f"总训练时间: {total_training_time/3600:.1f} 小时")
logger.info(f"最佳相关系数: {min_validation_loss:.6f} (Epoch {best_epoch+1})")
logger.info(f"最终相关系数: {corr:.6f}")

logger.info("开始生成训练结果图表...")
fig1, ax1 = plt.subplots()
ax1.plot(L_train)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title(f'Training Loss - Best Corr: {min_validation_loss:.4f}')

# 绘制验证损失函数折线图
fig2, ax2 = plt.subplots()
ax2.plot(L_val)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title(f'Validation Loss - Training Time: {total_training_time/3600:.1f}h')

fig3, ax3 = plt.subplots()
ax3.plot(AUC)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Test-set corr')
ax3.set_title(f'Correlation - Max: {min_validation_loss:.4f} @Epoch{best_epoch+1}')
ax3.axhline(y=min_validation_loss, color='r', linestyle='--', alpha=0.7)

fig1.savefig(f"./{MODEL}_Training Loss.png")
fig2.savefig(f"./{MODEL}_Validation Loss.png")
fig3.savefig(f"./{MODEL}_Test-set corr.png")

logger.info(f"结果图表已保存: {MODEL}_*.png")
logger.info(f"训练日志已保存: training.log")
logger.info(f"训练流程完成! - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
