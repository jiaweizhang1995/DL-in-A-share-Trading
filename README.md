# DL-in-A-share-Trading (GPU版本)

本项目是专为中国A股市场股票交易预测而设计的深度学习项目，采用GPU加速训练，支持大规模数据处理和高性能计算。

## 模型简介

本项目中的神经网络模型结合了多种深度学习技术，专为中国A股市场的股票交易预测而设计。模型架构融合了全连接层、1D 和 2D 卷积层、以及 LSTM（长短期记忆网络）层，能够处理复杂的时序数据和多维特征。模型的核心设计旨在捕捉股票市场的潜在模式，并生成精准的交易信号。

## 模型架构亮点

- **全连接层**：通过多个全连接层与批归一化和 Dropout 技术相结合，提高了模型的非线性表达能力，并有效防止过拟合
- **1D 卷积层**：提取时间序列数据中的局部特征，有效处理金融数据中的高频波动
- **2D 卷积层**：进一步挖掘二维特征间的相关性，为复杂的数据关系提供更深层次的洞察
- **LSTM 层**：利用双向4层 LSTM 处理序列数据中的长期依赖性，特别适合捕捉股市中的趋势和周期性变化
- **模型初始化**：采用 Kaiming 正态初始化，确保模型在训练初期具备良好的收敛性
- **GPU优化**：支持混合精度训练(FP16)、大批次处理、异步数据传输等GPU加速技术

## 系统要求

### 硬件要求
- **推荐**: NVIDIA GPU (8GB+ 显存)
- **内存**: 32GB+ 系统内存推荐 (64GB 最佳)
- **存储**: 10GB+ 可用空间

### 软件要求
- Python 3.8+
- CUDA 11.8 或 12.1
- cuDNN 8.7+

## 依赖库安装

### 1. CUDA 安装

首先安装CUDA工具包。推荐版本：

```bash
# CUDA 11.8 (推荐)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 或者 CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

### 2. Python 环境设置

推荐使用 conda 环境：

```bash
# 创建虚拟环境
conda create -n dl-trading python=3.10
conda activate dl-trading
```

### 3. 核心依赖安装

```bash
# PyTorch GPU版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或者 PyTorch GPU版本 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 科学计算库
pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0

# 可视化
pip install matplotlib==3.7.2 seaborn==0.12.2

# 进度条和工具
pip install tqdm==4.65.0

# 中国股市数据
pip install akshare==1.11.80 baostock==0.8.8

# GPU内存监控 (可选)
pip install pynvml==11.5.0 psutil==5.9.5

# HDF5支持 (用于大数据处理)
pip install h5py==3.9.0
```

### 4. 验证GPU安装

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

## 项目结构

```
DL-in-A-share-Trading/
├── Building_a_Dataset.py      # 数据采集脚本
├── Training_gpu.py            # GPU优化训练脚本
├── 5_fold_CV_gpu.py          # GPU优化5折交叉验证
├── APP.py                     # 模型应用脚本
├── resume_training.py         # 训练恢复工具
├── data/                      # 数据目录
├── weights/                   # 模型权重
├── checkpoints/              # 训练检查点
└── README.md
```

## 使用步骤

### 第一步：数据集构建

使用 baostock API 采集A股历史数据：

```bash
python Building_a_Dataset.py
```

这将在 `data/` 目录下生成CSV数据文件。

### 第二步：GPU训练

#### 基础训练
```bash
python Training_gpu.py
```

#### 完整GPU优化训练
```bash
python Training_gpu.py \
    --batch-size 1024 \
    --mixed-precision \
    --num-workers 8 \
    --epochs 50 \
    --gpu-id 0
```

#### 从检查点恢复训练
```bash
python Training_gpu.py \
    --resume ./checkpoints/training_checkpoint_epoch_20.pt \
    --batch-size 1024 \
    --mixed-precision
```

### 第三步：5折交叉验证

```bash
# 完整GPU优化的交叉验证
python 5_fold_CV_gpu.py \
    --batch-size 1024 \
    --mixed-precision \
    --num-workers 8
```

### 第四步：模型应用

完成训练后，使用模型进行预测：

```bash
python APP.py
```

## GPU优化特性

### 🚀 性能优化
- **混合精度训练**: 使用FP16减少显存使用，加速训练
- **大批次处理**: 默认1024批次大小，充分利用GPU并行能力
- **异步数据传输**: `pin_memory=True` + `non_blocking=True`
- **多线程数据加载**: 8个worker进程并行加载数据
- **CUDA后端优化**: 启用 `cudnn.benchmark` 自动优化

### 💾 内存管理
- **GPU内存监控**: 实时监控GPU内存使用情况
- **自动内存清理**: 定期清理GPU缓存
- **渐进式数据加载**: 避免一次性加载所有数据到内存
- **检查点系统**: 支持训练中断恢复，保护长时间训练

### 📊 监控和日志
- **详细的训练日志**: 记录GPU使用情况、内存状态
- **实时进度显示**: tqdm进度条显示训练和验证进度
- **性能指标追踪**: 损失函数、相关系数等指标的实时监控

## 命令行参数

### Training_gpu.py 参数
- `--batch-size`: 批次大小 (默认: 1024)
- `--gpu-id`: GPU设备ID (默认: 0)
- `--mixed-precision`: 启用混合精度训练
- `--num-workers`: 数据加载线程数 (默认: 8)
- `--epochs`: 训练轮数 (默认: 50)
- `--resume`: 从检查点恢复训练
- `--checkpoint-freq`: 检查点保存频率 (默认: 5轮)

### 5_fold_CV_gpu.py 参数
- 包含所有 Training_gpu.py 的参数
- `--resume-fold`: 从指定fold恢复交叉验证

## 性能基准

在NVIDIA RTX 4090 (24GB) 上的性能表现：

| 配置 | 批次大小 | 训练时间/轮 | 显存使用 | 吞吐量 |
|------|----------|-------------|----------|---------|
| FP32 | 512 | ~120s | ~18GB | ~66k samples/s |
| FP16 | 1024 | ~85s | ~12GB | ~94k samples/s |
| FP16 | 2048 | ~65s | ~20GB | ~123k samples/s |

## 故障排除

### CUDA相关问题
```bash
# 检查CUDA安装
nvidia-smi
nvcc --version

# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"
```

### 内存不足
- 降低 `--batch-size` 参数
- 启用 `--mixed-precision`
- 减少 `--num-workers`

### 性能优化建议
- 使用SSD存储数据文件
- 确保充足的系统内存 (32GB+)
- 监控GPU温度和功耗限制

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交 Issues 和 Pull Requests 来改进项目。

## 更新日志

### v2.0.0 (GPU版本)
- 完整GPU加速支持
- 混合精度训练
- 优化的数据加载管道
- 改进的检查点系统
- 清理项目结构，专注GPU优化