#!/usr/bin/env python3
"""
GPU环境检查脚本
验证深度学习训练所需的GPU环境是否正确配置
"""

import sys
import subprocess

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("=== 检查NVIDIA驱动 ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("✅ NVIDIA驱动正常")
        print(f"驱动信息:\n{result.stdout.split('Driver Version:')[1].split('CUDA Version:')[0].strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ NVIDIA驱动未安装或不可用")
        return False

def check_cuda():
    """检查CUDA安装"""
    print("\n=== 检查CUDA安装 ===")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
        print("✅ CUDA已安装")
        version_line = [line for line in result.stdout.split('\n') if 'release' in line][0]
        print(f"CUDA版本: {version_line.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ CUDA未安装或nvcc不在PATH中")
        return False

def check_pytorch():
    """检查PyTorch GPU支持"""
    print("\n=== 检查PyTorch GPU支持 ===")
    try:
        import torch
        print("✅ PyTorch已安装")
        print(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA支持正常")
            print(f"CUDA版本 (PyTorch): {torch.version.cuda}")
            print(f"可用GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  - 显存: {props.total_memory / 1024**3:.1f} GB")
                print(f"  - 计算能力: {props.major}.{props.minor}")
            return True
        else:
            print("❌ PyTorch无法使用CUDA")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def check_memory():
    """检查系统内存"""
    print("\n=== 检查系统内存 ===")
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / 1024**3
        available_gb = mem.available / 1024**3
        
        print(f"总内存: {total_gb:.1f} GB")
        print(f"可用内存: {available_gb:.1f} GB")
        
        if total_gb >= 32:
            print("✅ 内存充足 (≥32GB)")
            return True
        elif total_gb >= 16:
            print("⚠️  内存适中 (16-32GB) - 建议增加内存")
            return True
        else:
            print("❌ 内存不足 (<16GB) - 可能无法处理大型数据集")
            return False
    except ImportError:
        print("❌ psutil未安装，无法检查内存")
        return False

def check_dependencies():
    """检查其他依赖库"""
    print("\n=== 检查依赖库 ===")
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'matplotlib', 'tqdm', 
        'akshare', 'h5py'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + ' '.join(missing))
        return False
    else:
        print("✅ 所有依赖库已安装")
        return True

def test_gpu_training():
    """测试GPU训练能力"""
    print("\n=== GPU训练测试 ===")
    try:
        import torch
        import torch.nn as nn
        import time
        
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，跳过GPU测试")
            return False
        
        device = torch.device("cuda:0")
        print(f"使用设备: {device}")
        
        # 创建简单的测试模型
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        ).to(device)
        
        # 测试数据
        batch_size = 1024
        x = torch.randn(batch_size, 100).to(device)
        y = torch.randn(batch_size, 1).to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        # 热身
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # 测试性能
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(100):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        print(f"✅ GPU训练测试通过")
        print(f"平均训练时间: {avg_time:.2f} ms/batch")
        
        # 测试混合精度
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            
            with autocast():
                output = model(x)
                loss = criterion(output, y)
            
            print("✅ 混合精度训练支持正常")
            return True
        except ImportError:
            print("⚠️  混合精度训练不支持 (需要PyTorch >= 1.6)")
            return True
            
    except Exception as e:
        print(f"❌ GPU训练测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 深度学习GPU环境检查")
    print("=" * 50)
    
    checks = [
        check_nvidia_driver(),
        check_cuda(),
        check_pytorch(),
        check_memory(),
        check_dependencies(),
        test_gpu_training()
    ]
    
    passed = sum(checks)
    total = len(checks)
    
    print("\n" + "=" * 50)
    print(f"检查结果: {passed}/{total} 项通过")
    
    if passed == total:
        print("🎉 环境配置完美！可以开始GPU训练")
        return 0
    elif passed >= total - 1:
        print("⚠️  环境基本可用，但建议解决剩余问题")
        return 1
    else:
        print("❌ 环境配置有重大问题，请修复后再试")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)