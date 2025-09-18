"""
MLP配置示例文件
包含各种网络配置和超参数组合的示例
"""

from mlp_explorer import MLPExplorer

def basic_examples():
    """基础示例 - 演示不同数据类型和基本配置"""
    
    explorer = MLPExplorer()
    
    print("🎯 基础示例 - 不同数据类型的探索")
    print("=" * 60)
    
    # 配置1: 线性数据 - 简单网络
    print("\n📊 配置1: 线性数据 - 简单网络")
    result1 = explorer.explore_mlp(
        data_type='linear',
        hidden_sizes=[16],  # 单隐藏层，16个神经元
        learning_rate=0.01,
        epochs=100,
        activation='relu',
        noise_level=0.1
    )
    
    # 配置2: 二次数据 - 中等复杂度网络
    print("\n📊 配置2: 二次数据 - 中等复杂度网络")
    result2 = explorer.explore_mlp(
        data_type='quadratic',
        hidden_sizes=[32, 16],  # 两隐藏层
        learning_rate=0.01,
        epochs=150,
        activation='relu',
        noise_level=0.1
    )
    
    # 配置3: 正弦数据 - 更复杂网络
    print("\n📊 配置3: 正弦数据 - 更复杂网络")
    result3 = explorer.explore_mlp(
        data_type='sine',
        hidden_sizes=[64, 32, 16],  # 三隐藏层
        learning_rate=0.005,
        epochs=200,
        activation='tanh',  # tanh对周期性数据效果更好
        noise_level=0.1
    )
    
    # 配置4: 复杂数据 - 深度网络
    print("\n📊 配置4: 复杂数据 - 深度网络")
    result4 = explorer.explore_mlp(
        data_type='complex',
        hidden_sizes=[128, 64, 32, 16],  # 四隐藏层
        learning_rate=0.001,
        epochs=300,
        activation='relu',
        dropout_rate=0.1,  # 添加dropout防止过拟合
        noise_level=0.1
    )
    
    return [result1, result2, result3, result4]


def activation_comparison():
    """激活函数对比实验"""
    
    explorer = MLPExplorer()
    
    print("🧪 激活函数对比实验")
    print("=" * 60)
    
    activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
    results = {}
    
    for activation in activations:
        print(f"\n🔬 测试激活函数: {activation}")
        result = explorer.explore_mlp(
            data_type='sine',  # 使用正弦数据测试
            hidden_sizes=[64, 32],
            learning_rate=0.01,
            epochs=150,
            activation=activation,
            noise_level=0.1
        )
        results[activation] = result['metrics']
    
    # 比较结果
    print("\n📋 激活函数性能对比:")
    print("-" * 50)
    for activation, metrics in results.items():
        print(f"{activation:12s} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")
    
    return results


def learning_rate_tuning():
    """学习率调优实验"""
    
    explorer = MLPExplorer()
    
    print("⚡ 学习率调优实验")
    print("=" * 60)
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    results = {}
    
    for lr in learning_rates:
        print(f"\n📈 测试学习率: {lr}")
        result = explorer.explore_mlp(
            data_type='quadratic',
            hidden_sizes=[64, 32],
            learning_rate=lr,
            epochs=200,
            activation='relu',
            noise_level=0.1
        )
        results[lr] = result['metrics']
    
    # 比较结果
    print("\n📋 学习率性能对比:")
    print("-" * 50)
    for lr, metrics in results.items():
        print(f"LR {lr:8.4f} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")
    
    return results


def network_depth_experiment():
    """网络深度实验"""
    
    explorer = MLPExplorer()
    
    print("🏗️ 网络深度实验")
    print("=" * 60)
    
    # 不同深度的网络配置
    network_configs = [
        ([32], "浅层网络"),
        ([64, 32], "中等深度"),
        ([64, 32, 16], "较深网络"),
        ([128, 64, 32, 16], "深层网络"),
        ([128, 64, 32, 16, 8], "很深网络")
    ]
    
    results = {}
    
    for hidden_sizes, description in network_configs:
        print(f"\n🔧 测试 {description}: {hidden_sizes}")
        result = explorer.explore_mlp(
            data_type='complex',
            hidden_sizes=hidden_sizes,
            learning_rate=0.001,
            epochs=250,
            activation='relu',
            dropout_rate=0.1,  # 深层网络使用dropout
            noise_level=0.1
        )
        results[description] = result['metrics']
    
    # 比较结果
    print("\n📋 网络深度性能对比:")
    print("-" * 60)
    for desc, metrics in results.items():
        print(f"{desc:12s} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")
    
    return results


def optimizer_comparison():
    """优化器对比实验"""
    
    explorer = MLPExplorer()
    
    print("🏃‍♂️ 优化器对比实验")
    print("=" * 60)
    
    optimizers = ['adam', 'sgd', 'rmsprop']
    results = {}
    
    for optimizer in optimizers:
        print(f"\n⚙️ 测试优化器: {optimizer}")
        result = explorer.explore_mlp(
            data_type='quadratic',
            hidden_sizes=[64, 32],
            learning_rate=0.01,
            epochs=200,
            activation='relu',
            optimizer_type=optimizer,
            noise_level=0.1
        )
        results[optimizer] = result['metrics']
    
    # 比较结果
    print("\n📋 优化器性能对比:")
    print("-" * 50)
    for optimizer, metrics in results.items():
        print(f"{optimizer:10s} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")
    
    return results


def regularization_experiment():
    """正则化实验（Dropout效果）"""
    
    explorer = MLPExplorer()
    
    print("🛡️ 正则化实验 (Dropout)")
    print("=" * 60)
    
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    results = {}
    
    for dropout in dropout_rates:
        print(f"\n🎲 测试Dropout率: {dropout}")
        result = explorer.explore_mlp(
            data_type='complex',
            hidden_sizes=[128, 64, 32],
            learning_rate=0.001,
            epochs=250,
            activation='relu',
            dropout_rate=dropout,
            noise_level=0.1
        )
        results[dropout] = result['metrics']
    
    # 比较结果
    print("\n📋 Dropout效果对比:")
    print("-" * 50)
    for dropout, metrics in results.items():
        print(f"Dropout {dropout:.1f} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")
    
    return results


def comprehensive_tutorial():
    """综合教程 - 完整的调参流程"""
    
    print("🎓 MLP调参完整教程")
    print("=" * 80)
    
    print("""
    📚 本教程将带你了解MLP调参的完整流程:
    
    1. 📊 基础示例 - 了解不同数据类型和基本配置
    2. 🧪 激活函数选择 - 找到最适合的激活函数
    3. ⚡ 学习率调优 - 找到最佳学习率
    4. 🏗️ 网络深度优化 - 确定合适的网络层数
    5. 🏃‍♂️ 优化器选择 - 比较不同优化算法
    6. 🛡️ 正则化调节 - 防止过拟合
    
    每个步骤都会提供详细的解释和可视化结果。
    """)
    
    input("\n按回车键开始教程...")
    
    # 执行各个实验
    print("\n" + "="*80)
    basic_results = basic_examples()
    
    input("\n按回车键继续下一个实验...")
    print("\n" + "="*80)
    activation_results = activation_comparison()
    
    input("\n按回车键继续下一个实验...")
    print("\n" + "="*80)
    lr_results = learning_rate_tuning()
    
    input("\n按回车键继续下一个实验...")
    print("\n" + "="*80)
    depth_results = network_depth_experiment()
    
    input("\n按回车键继续下一个实验...")
    print("\n" + "="*80)
    optimizer_results = optimizer_comparison()
    
    input("\n按回车键继续最后一个实验...")
    print("\n" + "="*80)
    regularization_results = regularization_experiment()
    
    print("\n" + "="*80)
    print("🎉 恭喜! 你已经完成了MLP调参的完整教程!")
    print("💡 现在你可以根据学到的知识来调优自己的神经网络了!")
    
    return {
        'basic': basic_results,
        'activation': activation_results,
        'learning_rate': lr_results,
        'depth': depth_results,
        'optimizer': optimizer_results,
        'regularization': regularization_results
    }


if __name__ == "__main__":
    # 你可以运行以下任一函数来学习不同方面
    
    # 运行完整教程
    comprehensive_tutorial()
    
    # 或者单独运行某个实验
    # basic_examples()
    # activation_comparison()
    # learning_rate_tuning()
    # network_depth_experiment()
    # optimizer_comparison()
    # regularization_experiment()
