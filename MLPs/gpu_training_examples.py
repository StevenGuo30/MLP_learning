"""
GPU并行训练示例
展示如何使用GPU加速MLP训练，包括单GPU、多GPU数据并行等
"""

from mlp_explorer import MLPExplorer
from parallel_training import ParallelMLPExplorer
import torch
import time


def check_gpu_availability():
    """检查GPU可用性"""
    print("🔍 GPU环境检查")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.version.cuda}")
        print(f"🔥 GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # 显示当前GPU内存使用情况
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
        cached = torch.cuda.memory_reserved(current_device) / (1024**3)
        print(f"\n📊 当前GPU内存使用:")
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已缓存: {cached:.2f} GB")

        return True
    else:
        print("❌ CUDA不可用，将使用CPU训练")
        return False


def basic_gpu_training():
    """基础GPU训练示例"""
    print("\n🔥 基础GPU训练示例")
    print("=" * 50)

    explorer = MLPExplorer()

    # 使用GPU训练
    print("训练配置：使用GPU加速")
    result = explorer.explore_mlp(
        data_type="complex",
        hidden_sizes=[128, 64, 32],
        learning_rate=0.001,
        epochs=100,
        batch_size=64,  # 增大batch size以充分利用GPU
        use_gpu=True,
    )

    print(f"\n🎯 GPU训练结果:")
    print(f"R² 分数: {result['metrics']['r2']:.4f}")
    print(f"RMSE: {result['metrics']['rmse']:.4f}")

    return result


def compare_cpu_vs_gpu():
    """CPU vs GPU性能对比"""
    print("\n⚔️ CPU vs GPU性能对比")
    print("=" * 50)

    explorer = MLPExplorer()

    # 相同的训练配置
    config = {
        "data_type": "complex",
        "hidden_sizes": [256, 128, 64, 32],
        "learning_rate": 0.001,
        "epochs": 50,  # 减少epochs以便快速对比
        "batch_size": 64,
        "n_samples": 5000,
    }

    # CPU训练
    print("🖥️ CPU训练中...")
    start_time = time.time()
    result_cpu = explorer.explore_mlp(**config, use_gpu=False)
    cpu_time = time.time() - start_time

    # GPU训练（如果可用）
    if torch.cuda.is_available():
        print("\n🔥 GPU训练中...")
        start_time = time.time()
        result_gpu = explorer.explore_mlp(**config, use_gpu=True)
        gpu_time = time.time() - start_time

        # 性能对比
        print(f"\n📊 性能对比结果:")
        print("-" * 40)
        print(f"CPU训练时间: {cpu_time:.2f}秒")
        print(f"GPU训练时间: {gpu_time:.2f}秒")
        print(f"GPU加速比: {cpu_time/gpu_time:.2f}x")
        print(f"\nCPU最终R²: {result_cpu['metrics']['r2']:.4f}")
        print(f"GPU最终R²: {result_gpu['metrics']['r2']:.4f}")

        return {"cpu": result_cpu, "gpu": result_gpu, "speedup": cpu_time / gpu_time}
    else:
        print("⚠️ GPU不可用，无法进行对比")
        return {"cpu": result_cpu}


def multi_gpu_training():
    """多GPU数据并行训练示例"""
    print("\n🚀 多GPU数据并行训练")
    print("=" * 50)

    if torch.cuda.device_count() < 2:
        print("⚠️ 需要至少2个GPU才能进行多GPU训练")
        print(f"当前可用GPU数量: {torch.cuda.device_count()}")
        return

    # 使用并行训练探索器
    parallel_explorer = ParallelMLPExplorer()

    print(f"🎯 使用 {torch.cuda.device_count()} 个GPU进行数据并行训练")

    # 多GPU训练对比
    results = parallel_explorer.compare_training_methods(
        data_type="complex",
        hidden_sizes=[256, 128, 64, 32],
        epochs=100,
        batch_size=128,  # 更大的batch size以充分利用多GPU
        n_samples=10000,
    )

    return results


def memory_efficient_training():
    """显存优化训练示例"""
    print("\n🧠 显存优化训练技巧")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("⚠️ 需要GPU支持")
        return

    explorer = MLPExplorer()

    # 检查初始显存使用
    initial_memory = torch.cuda.memory_allocated() / (1024**3)
    print(f"初始显存使用: {initial_memory:.2f} GB")

    print("\n1️⃣ 大batch size训练（可能导致显存不足）")
    try:
        result_large = explorer.explore_mlp(
            data_type="complex",
            hidden_sizes=[512, 256, 128, 64],
            epochs=20,
            batch_size=256,  # 大batch size
            n_samples=10000,
            use_gpu=True,
        )
        print("✅ 大batch size训练成功")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ 显存不足，训练失败")
        else:
            print(f"❌ 其他错误: {e}")

    # 清理显存
    torch.cuda.empty_cache()

    print("\n2️⃣ 优化后的训练配置")
    result_optimized = explorer.explore_mlp(
        data_type="complex",
        hidden_sizes=[256, 128, 64, 32],
        epochs=50,
        batch_size=64,  # 适中的batch size
        n_samples=10000,
        use_gpu=True,
    )

    final_memory = torch.cuda.memory_allocated() / (1024**3)
    print(f"\n📊 显存使用情况:")
    print(f"最终显存使用: {final_memory:.2f} GB")
    print(f"显存增加: {final_memory - initial_memory:.2f} GB")

    return result_optimized


def mixed_precision_demo():
    """混合精度训练演示"""
    print("\n⚡ 混合精度训练演示")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("⚠️ 需要GPU支持混合精度训练")
        return

    # 检查是否支持自动混合精度
    if hasattr(torch.cuda, "amp"):
        print("✅ 支持自动混合精度(AMP)")

        # 使用并行训练探索器的混合精度功能
        parallel_explorer = ParallelMLPExplorer()
        parallel_explorer.demonstrate_mixed_precision_training(
            data_type="complex", hidden_sizes=[512, 256, 128, 64], epochs=100
        )
    else:
        print("❌ 当前PyTorch版本不支持自动混合精度")


def gpu_training_best_practices():
    """GPU训练最佳实践指南"""
    print("\n💡 GPU训练最佳实践")
    print("=" * 80)

    print(
        """
    🎯 GPU训练优化建议:
    
    1. 📊 批次大小(Batch Size)优化:
       - GPU: 64-256 (根据显存调整)
       - CPU: 16-64
       - 更大的batch size通常能更好地利用GPU并行性
    
    2. 🏗️ 网络结构设计:
       - GPU适合较大的网络（更多参数）
       - 利用GPU的并行计算能力
       - 避免过小的网络（GPU利用率低）
    
    3. 💾 数据加载优化:
       - 使用pin_memory=True加速数据传输
       - 使用non_blocking=True异步数据传输
       - 适当的num_workers数量
    
    4. 🔄 显存管理:
       - 使用torch.cuda.empty_cache()清理显存
       - 监控显存使用情况
       - 必要时减小batch size或网络大小
    
    5. ⚡ 混合精度训练:
       - 使用torch.cuda.amp.autocast()
       - 可以节省约50%显存
       - 训练速度提升20-50%
    
    6. 🚀 多GPU策略:
       - nn.DataParallel: 简单易用，适合小规模
       - DistributedDataParallel: 更高效，适合大规模训练
       - 注意负载均衡
    """
    )


def run_all_gpu_examples():
    """运行所有GPU训练示例"""
    print("🎓 GPU并行训练完整教程")
    print("=" * 80)

    # 1. 检查GPU环境
    gpu_available = check_gpu_availability()

    if not gpu_available:
        print("\n⚠️ 没有可用的GPU，部分示例将跳过")
        return

    # 2. 基础GPU训练
    print("\n" + "=" * 60)
    basic_gpu_training()

    # 3. CPU vs GPU对比
    print("\n" + "=" * 60)
    compare_cpu_vs_gpu()

    # 4. 多GPU训练
    print("\n" + "=" * 60)
    multi_gpu_training()

    # 5. 显存优化
    print("\n" + "=" * 60)
    memory_efficient_training()

    # 6. 混合精度训练
    print("\n" + "=" * 60)
    mixed_precision_demo()

    # 7. 最佳实践
    print("\n" + "=" * 60)
    gpu_training_best_practices()

    print("\n🎉 GPU训练教程完成!")
    print("🚀 现在你已经掌握了GPU加速训练的各种技巧!")


if __name__ == "__main__":
    # 运行完整的GPU训练教程
    run_all_gpu_examples()

    # 或者单独运行某个示例
    # check_gpu_availability()
    # basic_gpu_training()
    # compare_cpu_vs_gpu()
    # multi_gpu_training()
    # memory_efficient_training()
    # mixed_precision_demo()
