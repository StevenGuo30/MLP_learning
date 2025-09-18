# 🎓 MLP学习项目：探究数据关系与神经网络调参

本项目是一个完整的多层感知机(MLP)学习教程，通过探究两组数据之间的关系来学习神经网络的配置和调参技巧。项目包含可视化功能，让你直观地理解训练过程和模型性能。

## 🌟 项目特色

- 📊 **多种数据关系**: 线性、二次、正弦、复杂非线性关系
- 🧠 **可配置网络结构**: 灵活设置层数、神经元数量、激活函数等
- ⚙️ **全面超参数调优**: 学习率、优化器、批次大小、正则化等
- 📈 **丰富可视化**: 训练历史、预测结果、残差分析等
- 🔥 **GPU加速训练**: 单GPU、多GPU数据并行、混合精度训练
- 🚀 **分布式训练**: 支持多机多卡分布式训练
- 🎮 **交互式学习**: Jupyter notebook和Python脚本两种方式
- 📚 **详细教程**: 从基础到高级的完整学习路径

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
cd MLPs

# 安装依赖
pip install -r ../requirements.txt
```

### 2. 运行快速示例

```python
# 最简单的使用方式
from mlp_explorer import MLPExplorer

explorer = MLPExplorer()
result = explorer.explore_mlp()
```

### 3. 运行交互式教程

```bash
# 运行快速开始脚本
python quick_start.py

# 或启动Jupyter notebook
jupyter notebook mlp_learning_notebook.ipynb
```

## 📁 项目结构

```
MLPs/
├── mlp_explorer.py          # 核心MLP实现（支持GPU加速）
├── parallel_training.py     # 并行训练实现（数据并行、分布式训练）
├── gpu_training_examples.py # GPU训练示例和最佳实践
├── config_examples.py       # 配置示例和实验
├── quick_start.py          # 快速开始脚本
├── mlp_learning_notebook.ipynb  # 交互式学习notebook
└── README.md               # 项目说明
```

## 🧠 核心功能详解

### 1. 数据生成器 (DataGenerator)

支持四种数据关系类型：

```python
# 线性关系: y = 3x + 2 + noise
x, y = DataGenerator.generate_linear_data(1000, 0.1)

# 二次关系: y = 2x² - 3x + 1 + noise  
x, y = DataGenerator.generate_quadratic_data(1000, 0.1)

# 正弦关系: y = sin(x) + 0.5cos(2x) + noise
x, y = DataGenerator.generate_sine_data(1000, 0.1)

# 复杂关系: 多种函数的组合
x, y = DataGenerator.generate_complex_data(1000, 0.1)
```

### 2. MLP网络结构 (MLPRegressor)

高度可配置的神经网络：

```python
model = MLPRegressor(
    input_size=1,              # 输入维度
    hidden_sizes=[64, 32, 16], # 隐藏层配置
    output_size=1,             # 输出维度
    dropout_rate=0.1,          # Dropout率
    activation='relu'          # 激活函数
)
```

支持的激活函数：
- `relu`: 最常用，适合大多数情况
- `tanh`: 适合周期性数据
- `sigmoid`: 输出需要在(0,1)区间时
- `leaky_relu`: 缓解ReLU的"死神经元"问题
- `elu`: 更平滑的激活函数

### 3. 训练器 (MLPTrainer)

支持多种优化器和训练策略：

```python
trainer = MLPTrainer(
    model=model,
    learning_rate=0.01,
    optimizer_type='adam'  # 'adam', 'sgd', 'rmsprop'
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=200,
    verbose=True
)
```

### 4. 可视化器 (MLPVisualizer)

提供丰富的可视化功能：

- 数据关系图
- 训练历史曲线
- 预测结果对比
- 残差分析
- 性能指标展示

## ⚙️ 参数配置指南

### 网络结构设计

```python
# 简单数据 -> 浅层网络
hidden_sizes = [32]           # 单层

# 中等复杂度 -> 中等深度
hidden_sizes = [64, 32]       # 两层

# 复杂数据 -> 深层网络  
hidden_sizes = [128, 64, 32, 16]  # 四层
```

### 学习率选择

```python
# 大学习率：快速收敛但可能不稳定
learning_rate = 0.1

# 中等学习率：平衡收敛速度和稳定性  
learning_rate = 0.01          # 推荐起始值

# 小学习率：稳定但收敛慢
learning_rate = 0.001
```

### 正则化配置

```python
# 无正则化
dropout_rate = 0.0

# 轻度正则化
dropout_rate = 0.1            # 推荐值

# 强正则化
dropout_rate = 0.3
```

## 🎯 实验示例

### 基础实验

```python
from mlp_explorer import MLPExplorer

explorer = MLPExplorer()

# 实验1: 线性数据 + 简单网络
result1 = explorer.explore_mlp(
    data_type='linear',
    hidden_sizes=[16],
    learning_rate=0.01,
    epochs=100
)

# 实验2: 复杂数据 + 深层网络
result2 = explorer.explore_mlp(
    data_type='complex',
    hidden_sizes=[128, 64, 32, 16],
    learning_rate=0.001,
    epochs=300,
    dropout_rate=0.1
)
```

### 对比实验

```python
# 激活函数对比
activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
for activation in activations:
    result = explorer.explore_mlp(
        data_type='sine',
        hidden_sizes=[64, 32],
        activation=activation
    )
    print(f"{activation}: R² = {result['metrics']['r2']:.4f}")
```

### 调参实验

```python
# 学习率调优
learning_rates = [0.1, 0.01, 0.001, 0.0001]
for lr in learning_rates:
    result = explorer.explore_mlp(
        data_type='quadratic',
        learning_rate=lr
    )
    print(f"LR {lr}: R² = {result['metrics']['r2']:.4f}")
```

## 📊 性能评估指标

项目提供三个主要评估指标：

1. **均方误差 (MSE)**: 平均预测误差的平方
2. **均方根误差 (RMSE)**: MSE的平方根，与目标变量同单位
3. **决定系数 (R²)**: 解释方差比例，越接近1越好

```python
metrics = result['metrics']
print(f"MSE: {metrics['mse']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")  
print(f"R²: {metrics['r2']:.4f}")
```

## 🔥 GPU并行训练

### GPU训练基础使用

```python
from mlp_explorer import MLPExplorer

explorer = MLPExplorer()

# 启用GPU训练（自动检测并使用可用GPU）
result = explorer.explore_mlp(
    data_type='complex',
    hidden_sizes=[256, 128, 64],
    use_gpu=True,  # 启用GPU加速
    batch_size=128  # GPU可以使用更大的batch size
)
```

### 多GPU数据并行

```python
from parallel_training import ParallelMLPExplorer

parallel_explorer = ParallelMLPExplorer()

# 自动使用所有可用GPU进行数据并行训练
results = parallel_explorer.compare_training_methods(
    data_type='complex',
    hidden_sizes=[512, 256, 128, 64],
    epochs=200,
    batch_size=256,  # 多GPU可以使用更大的batch size
    n_samples=20000
)
```

### GPU训练优化技巧

1. **批次大小优化**:
   - CPU: 16-64
   - 单GPU: 64-256
   - 多GPU: 128-512

2. **显存管理**:
   ```python
   # 清理显存
   torch.cuda.empty_cache()
   
   # 监控显存使用
   allocated = torch.cuda.memory_allocated() / (1024**3)
   print(f"显存使用: {allocated:.2f} GB")
   ```

3. **混合精度训练** (节省50%显存，提升20-50%速度):
   ```python
   parallel_explorer.demonstrate_mixed_precision_training()
   ```

### 分布式训练

```python
# 多机多卡分布式训练
from parallel_training import run_distributed_training_example
run_distributed_training_example()
```

## 💡 最佳实践建议

### 1. 数据预处理
- ✅ 总是标准化输入数据
- ✅ 合理划分训练/验证集
- ✅ 适当的噪声水平测试鲁棒性

### 2. 网络设计  
- ✅ 从简单结构开始
- ✅ 根据数据复杂度调整网络容量
- ✅ 避免过度复杂的网络结构

### 3. 训练策略
- ✅ 监控训练和验证损失
- ✅ 使用早停避免过拟合
- ✅ 多次运行取平均结果

### 4. 超参数调优
- ✅ 学习率从0.01开始尝试
- ✅ 批次大小: CPU(32-64), GPU(64-256)
- ✅ 深层网络使用适当的dropout

### 5. GPU训练优化
- ✅ 使用pin_memory=True加速数据传输
- ✅ 合理设置batch_size充分利用GPU
- ✅ 监控显存使用，避免OOM错误
- ✅ 考虑使用混合精度训练节省显存

## 🔍 常见问题解答

### Q: 为什么模型性能很差？
A: 检查以下几点：
- 网络容量是否匹配数据复杂度
- 学习率是否合适
- 训练轮数是否足够
- 是否有过拟合现象

### Q: 如何选择网络结构？
A: 基本原则：
- 简单数据：1-2层隐藏层
- 复杂数据：2-4层隐藏层
- 每层神经元数量通常递减

### Q: 训练过程震荡怎么办？
A: 可能的解决方案：
- 降低学习率
- 使用批量归一化
- 增加正则化
- 检查数据质量

## 🎮 交互式学习

### Jupyter Notebook
打开 `mlp_learning_notebook.ipynb` 获得最佳学习体验：
- 📊 可视化数据关系
- 🧪 交互式参数调节
- 📈 实时结果展示
- 💡 详细学习指导

### Python脚本
运行不同的学习模块：

```bash
# 快速开始
python quick_start.py

# 配置示例
python config_examples.py

# GPU并行训练示例
python gpu_training_examples.py

# 高级并行训练
python parallel_training.py

# 自定义实验
python -c "from mlp_explorer import MLPExplorer; explorer = MLPExplorer(); explorer.explore_mlp()"
```

## 🎓 学习路径推荐

### 初学者路径
1. 运行 `quick_start.py` 了解基本概念
2. 阅读代码理解实现原理
3. 尝试修改参数观察效果
4. 学习可视化结果的解读

### 进阶路径  
1. 完成 `config_examples.py` 中的所有实验
2. 理解不同激活函数的特性
3. 掌握超参数调优技巧
4. 尝试自定义数据类型

### 高级路径
1. 学习GPU并行训练 (`gpu_training_examples.py`)
2. 掌握多GPU数据并行和分布式训练
3. 修改网络结构代码
4. 实现新的激活函数和优化技术
5. 添加其他正则化技术
6. 扩展到多维输入输出

## 📚 参考资源

- [PyTorch官方文档](https://pytorch.org/docs/)
- [Deep Learning - Ian Goodfellow](https://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

## 🤝 贡献指南

欢迎提出改进建议：
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

MIT License - 详见LICENSE文件

---

🎉 **开始你的MLP学习之旅吧！** 

如果你觉得这个项目有帮助，请给它一个⭐️！
