#!/usr/bin/env python3
"""
快速恢复训练脚本
用于在训练意外中断后快速恢复
"""

import os
import glob
import subprocess
import sys
from datetime import datetime

def find_latest_checkpoint():
    """查找最新的checkpoint文件"""
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        print("❌ 没有找到 checkpoints 目录")
        return None
    
    # 查找最新的checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    if os.path.exists(latest_checkpoint):
        print(f"🔍 找到最新checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    # 如果没有latest，查找编号最大的checkpoint
    pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')
    checkpoints = glob.glob(pattern)
    if checkpoints:
        # 按epoch编号排序
        checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
        latest = checkpoints[-1]
        print(f"🔍 找到最新编号checkpoint: {latest}")
        return latest
    
    print("❌ 没有找到任何checkpoint文件")
    return None

def list_all_checkpoints():
    """列出所有可用的checkpoint"""
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        print("❌ 没有找到 checkpoints 目录")
        return []
    
    checkpoints = []
    
    # latest checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    if os.path.exists(latest_checkpoint):
        stat = os.stat(latest_checkpoint)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        checkpoints.append(('最新checkpoint', latest_checkpoint, mtime))
    
    # 编号checkpoint
    pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')
    epoch_checkpoints = glob.glob(pattern)
    for cp in sorted(epoch_checkpoints, key=lambda x: int(x.split('_epoch_')[1].split('.')[0])):
        epoch_num = cp.split('_epoch_')[1].split('.')[0]
        stat = os.stat(cp)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        checkpoints.append((f'Epoch {epoch_num}', cp, mtime))
    
    return checkpoints

def main():
    print("🚀 训练恢复助手")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        print("📋 可用的checkpoint文件:")
        checkpoints = list_all_checkpoints()
        if not checkpoints:
            print("❌ 没有找到任何checkpoint")
            return
        
        for i, (name, path, mtime) in enumerate(checkpoints, 1):
            print(f"{i:2d}. {name:15s} | {mtime} | {path}")
        
        print("\\n💡 使用方法:")
        print("   python resume_training.py                    # 自动从最新checkpoint恢复")
        print("   python Training.py --resume <checkpoint路径>  # 手动指定checkpoint")
        return
    
    # 自动查找最新checkpoint并恢复训练
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        print("\\n💡 建议:")
        print("   1. 确保之前的训练已经保存了checkpoint")
        print("   2. 检查 ./checkpoints/ 目录")
        print("   3. 使用 'python resume_training.py --list' 查看所有checkpoint")
        return
    
    print(f"\\n🔄 开始从checkpoint恢复训练...")
    print(f"📁 Checkpoint路径: {checkpoint_path}")
    
    # 构建恢复命令
    cmd = ['python', 'Training.py', '--resume', checkpoint_path]
    
    print(f"🏃‍♂️ 执行命令: {' '.join(cmd)}")
    print("=" * 50)
    
    # 执行恢复训练
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\\n⏹️  训练被用户中断")
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ 训练执行失败: {e}")
    except Exception as e:
        print(f"\\n❌ 发生错误: {e}")

if __name__ == '__main__':
    main()