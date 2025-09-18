"""
MLP快速开始脚本
一个简单的示例，展示如何使用MLP探索器
"""

from mlp_explorer import MLPExplorer


def quick_demo():
    """快速演示MLP的基本用法"""

    print("🎯 MLP快速演示")
    print("=" * 50)

    # 创建探索器
    explorer = MLPExplorer()

    # 快速示例：使用二次数据
    print("\n📊 示例：学习二次函数关系")
    print("数据: y = 2x² - 3x + 1 + noise")

    result = explorer.explore_mlp(
        data_type="quadratic",  # 二次关系数据
        hidden_sizes=[64, 32],  # 两个隐藏层：64和32个神经元
        learning_rate=0.01,  # 学习率
        epochs=150,  # 训练150轮
        activation="relu",  # ReLU激活函数
        noise_level=0.1,  # 10%噪声
    )

    print(f"\n🎉 训练完成!")
    print(f"R² 分数: {result['metrics']['r2']:.4f}")
    print(f"RMSE: {result['metrics']['rmse']:.4f}")

    if result["metrics"]["r2"] > 0.9:
        print("🏆 优秀! 模型学到了数据的关系")
    elif result["metrics"]["r2"] > 0.8:
        print("👍 不错! 模型表现良好")
    else:
        print("📈 可以尝试调整参数以提高性能")

    return result


def parameter_examples():
    """展示不同参数配置的例子"""

    print("\n\n🔧 参数配置示例")
    print("=" * 50)

    explorer = MLPExplorer()

    # 示例配置
    configs = [
        {
            "name": "简单线性网络",
            "data_type": "linear",
            "hidden_sizes": [16],
            "learning_rate": 0.01,
            "epochs": 100,
            "activation": "relu",
        },
        {
            "name": "复杂非线性网络",
            "data_type": "complex",
            "hidden_sizes": [128, 64, 32],
            "learning_rate": 0.001,
            "epochs": 200,
            "activation": "relu",
            "dropout_rate": 0.1,
        },
        {
            "name": "正弦数据专用网络",
            "data_type": "sine",
            "hidden_sizes": [64, 32],
            "learning_rate": 0.005,
            "epochs": 200,
            "activation": "tanh",  # tanh对周期性数据效果更好
        },
    ]

    results = {}

    for config in configs:
        name = config.pop("name")
        print(f"\n🧪 测试配置: {name}")

        result = explorer.explore_mlp(**config)
        results[name] = result["metrics"]

        print(
            f"R²: {result['metrics']['r2']:.4f}, RMSE: {result['metrics']['rmse']:.4f}"
        )

    # 总结
    print(f"\n📋 配置性能总结:")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:20s} | R²: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f}")

    return results


def interactive_learning():
    """交互式学习模式"""

    print("\n\n🎮 交互式学习模式")
    print("=" * 50)
    print("让我们一步步构建和训练MLP!")

    explorer = MLPExplorer()

    # 数据类型选择
    print("\n📊 第一步：选择数据类型")
    print("1. linear - 线性关系 (y = 3x + 2)")
    print("2. quadratic - 二次关系 (y = 2x² - 3x + 1)")
    print("3. sine - 正弦关系 (y = sin(x) + 0.5cos(2x))")
    print("4. complex - 复杂关系 (多种函数组合)")

    data_choice = input("\n选择数据类型 (1-4, 默认2): ").strip()
    data_types = {"1": "linear", "2": "quadratic", "3": "sine", "4": "complex"}
    data_type = data_types.get(data_choice, "quadratic")

    # 网络结构
    print(f"\n🧠 第二步：设计网络结构")
    print("网络层数建议:")
    print("- 简单数据: 1-2层")
    print("- 复杂数据: 2-4层")

    layers_input = input("输入隐藏层大小 (例如: 64,32 表示两层，默认: 64,32): ").strip()
    if layers_input:
        try:
            hidden_sizes = [int(x.strip()) for x in layers_input.split(",")]
        except:
            hidden_sizes = [64, 32]
    else:
        hidden_sizes = [64, 32]

    # 学习率
    lr_input = input("\n⚡ 第三步：设置学习率 (推荐 0.001-0.01, 默认: 0.01): ").strip()
    try:
        learning_rate = float(lr_input) if lr_input else 0.01
    except:
        learning_rate = 0.01

    # 训练轮数
    epochs_input = input(
        "\n🔄 第四步：设置训练轮数 (推荐 100-300, 默认: 150): "
    ).strip()
    try:
        epochs = int(epochs_input) if epochs_input else 150
    except:
        epochs = 150

    print(f"\n🚀 开始训练...")
    print(
        f"配置: 数据={data_type}, 网络={hidden_sizes}, 学习率={learning_rate}, 轮数={epochs}"
    )

    result = explorer.explore_mlp(
        data_type=data_type,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        epochs=epochs,
        activation="relu",
    )

    print(f"\n🎊 训练完成!")
    print(f"最终性能: R² = {result['metrics']['r2']:.4f}")

    # 给出建议
    r2 = result["metrics"]["r2"]
    if r2 > 0.95:
        print("🏆 优秀! 这是一个很好的配置!")
    elif r2 > 0.9:
        print("👍 不错! 可以尝试微调参数进一步优化")
    elif r2 > 0.8:
        print("📈 还可以! 建议:")
        print("  - 增加隐藏层大小或层数")
        print("  - 调整学习率")
        print("  - 增加训练轮数")
    else:
        print("🔧 需要改进! 建议:")
        print("  - 检查数据复杂度和网络容量是否匹配")
        print("  - 尝试不同的学习率")
        print("  - 增加网络深度或宽度")

    return result


if __name__ == "__main__":
    print("🎓 MLP学习快速开始指南")
    print("=" * 80)

    # 运行快速演示
    quick_demo()

    # 运行参数示例
    parameter_examples()

    # 交互式学习（注释掉以避免在自动运行时等待输入）
    # interactive_learning()

    print("\n\n🎉 快速开始完成!")
    print("💡 下一步:")
    print("  1. 运行 config_examples.py 查看更多配置示例")
    print("  2. 打开 mlp_learning_notebook.ipynb 进行交互式学习")
    print("  3. 阅读 README.md 了解详细使用说明")
