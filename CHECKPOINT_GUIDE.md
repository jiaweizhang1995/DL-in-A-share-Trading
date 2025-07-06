# 训练断点续训指南

本指南详细说明如何使用新增的checkpoint功能来实现训练的断点续训。

## 🚀 功能特性

- **自动checkpoint保存**: 每N个epoch自动保存训练状态
- **最新状态保存**: 每个epoch都保存最新状态
- **完整状态恢复**: 包括模型权重、优化器状态、调度器状态、训练历史等
- **数据缓存**: 预处理数据自动缓存，恢复时无需重新处理
- **灵活的恢复选项**: 支持从任意checkpoint恢复

## 📁 文件结构

```
./checkpoints/                     # checkpoint目录
├── checkpoint_latest.pth          # 最新的训练状态（每个epoch更新）
├── checkpoint_epoch_5.pth         # 第5个epoch的checkpoint
├── checkpoint_epoch_10.pth        # 第10个epoch的checkpoint
└── ...

./processed_data.npz               # 缓存的预处理数据
./training.log                     # 训练日志
```

## 🔧 使用方法

### 1. 正常训练（支持checkpoint）

```bash
# 从头开始训练，每5个epoch保存一次checkpoint
python Training.py

# 自定义checkpoint保存频率（每3个epoch保存一次）
python Training.py --checkpoint-freq 3
```

### 2. 恢复训练的几种方式

#### 方式一：自动恢复（推荐）
```bash
# 使用快速恢复脚本，自动找到最新checkpoint
python resume_training.py
```

#### 方式二：手动指定checkpoint
```bash
# 从最新checkpoint恢复
python Training.py --resume ./checkpoints/checkpoint_latest.pth

# 从特定epoch的checkpoint恢复
python Training.py --resume ./checkpoints/checkpoint_epoch_10.pth
```

#### 方式三：查看所有可用checkpoint
```bash
# 列出所有可用的checkpoint文件
python resume_training.py --list
```

### 3. 训练意外中断后的恢复步骤

1. **检查checkpoint状态**：
   ```bash
   python resume_training.py --list
   ```

2. **选择恢复方式**：
   - 如果要从最新状态恢复：`python resume_training.py`
   - 如果要从特定epoch恢复：`python Training.py --resume ./checkpoints/checkpoint_epoch_X.pth`

3. **验证恢复成功**：
   - 检查日志中的"恢复训练，从 epoch X 开始"信息
   - 确认当前最佳相关系数正确显示

## 📊 checkpoint包含的信息

每个checkpoint文件包含：
- `epoch`: 当前epoch编号
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `scheduler_state_dict`: 学习率调度器状态
- `train_losses`: 训练损失历史
- `val_losses`: 验证损失历史
- `aucs`: 相关系数历史
- `best_corr`: 最佳相关系数
- `best_epoch`: 最佳epoch编号
- `timestamp`: 保存时间戳

## ⚠️ 注意事项

1. **数据一致性**: 恢复训练时确保使用相同的数据文件
2. **环境一致性**: 保持Python环境和依赖版本一致
3. **磁盘空间**: checkpoint文件较大，注意磁盘空间
4. **定期清理**: 可以删除较早的checkpoint文件以节省空间

## 🔍 故障排除

### 问题1: 找不到checkpoint文件
```bash
# 检查checkpoints目录
ls -la ./checkpoints/

# 检查是否有预处理数据缓存
ls -la ./processed_data.npz
```

### 问题2: checkpoint加载失败
- 确认checkpoint文件没有损坏
- 检查Python环境和依赖版本
- 查看训练日志中的错误信息

### 问题3: 恢复后性能不一致
- 确认使用相同的数据文件
- 检查随机种子设置
- 验证模型架构没有变化

## 💡 最佳实践

1. **定期保存**: 使用合适的checkpoint频率（推荐3-5个epoch）
2. **监控磁盘**: 定期清理旧的checkpoint文件
3. **备份重要checkpoint**: 将关键的checkpoint文件备份到其他位置
4. **记录训练参数**: 在日志中记录重要的训练参数以便复现

## 📝 示例场景

### 场景1: 训练在第15个epoch意外中断
```bash
# 1. 查看可用checkpoint
python resume_training.py --list

# 2. 自动恢复（从最新状态继续）
python resume_training.py

# 训练将从第16个epoch开始继续
```

### 场景2: 想从第10个epoch重新训练
```bash
# 从特定checkpoint恢复
python Training.py --resume ./checkpoints/checkpoint_epoch_10.pth

# 训练将从第11个epoch开始
```

### 场景3: 更改checkpoint保存频率
```bash
# 每2个epoch保存一次（更频繁）
python Training.py --checkpoint-freq 2

# 每10个epoch保存一次（节省空间）
python Training.py --checkpoint-freq 10
```

---

通过这个checkpoint系统，你可以安全地进行长时间训练，不用担心意外中断导致的训练进度丢失！