# 🚀 PyTorch MLP并行训练完整指南

本指南详细介绍了使用PyTorch实现MLP并行训练的各种方法，从基础的GPU加速到高级的分布式训练。

## 📋 目录

1. [基础GPU训练](#1-基础gpu训练)
2. [数据并行 (DataParallel)](#2-数据并行-dataparallel)
3. [分布式数据并行 (DistributedDataParallel)](#3-分布式数据并行-distributeddataparallel)
4. [混合精度训练](#4-混合精度训练)
5. [模型并行](#5-模型并行)
6. [性能优化技巧](#6-性能优化技巧)
7. [最佳实践](#7-最佳实践)

## 1. 基础GPU训练

### 1.1 单GPU训练

最简单的GPU加速方法，将模型和数据移动到GPU：

```python
import torch
import torch.nn as nn

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 将模型移到GPU
model = MLPRegressor(input_size=1, hidden_sizes=[64, 32], output_size=1)
model = model.to(device)

# 训练循环中将数据移到GPU
for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device, non_blocking=True)  # non_blocking异步传输
    batch_y = batch_y.to(device, non_blocking=True)
    
    # 前向传播
    outputs = model(batch_x)
    loss = criterion(outputs.squeeze(), batch_y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 1.2 数据加载优化

```python
# 优化数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=128,  # GPU可以使用更大的batch size
    shuffle=True,
    pin_memory=True,     # 加速CPU到GPU的数据传输
    num_workers=4        # 多进程数据加载
)
```

## 2. 数据并行 (DataParallel)

### 2.1 基本用法

适用于单机多GPU的情况，使用简单但效率相对较低：

```python
import torch.nn as nn

# 检查GPU数量
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行训练")
    model = nn.DataParallel(model)

model = model.to(device)

# 训练代码不需要修改
for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    outputs = model(batch_x)
    loss = criterion(outputs.squeeze(), batch_y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 2.2 DataParallel的特点

**优点**：
- 使用简单，代码修改最少
- 自动处理数据分发和梯度聚合

**缺点**：
- 主GPU负载较重（梯度聚合和参数更新）
- GPU间通信开销大
- 扩展性有限

## 3. 分布式数据并行 (DistributedDataParallel)

### 3.1 单机多GPU (推荐)

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def train_distributed(rank, world_size):
    # 设置分布式环境
    setup(rank, world_size)
    
    # 设置GPU设备
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # 创建模型并移到对应GPU
    model = MLPRegressor(input_size=1, hidden_sizes=[64, 32], output_size=1)
    model = model.to(device)
    
    # 包装为DDP模型
    model = DDP(model, device_ids=[rank])
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        pin_memory=True
    )
    
    # 训练循环
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # 重要：确保每个epoch数据打乱
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    cleanup()

# 启动多进程训练
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
```

### 3.2 多机多GPU

```python
def setup_multi_node(rank, world_size, master_addr, master_port):
    """多机分布式训练设置"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 机器1运行
python train.py --rank 0 --world-size 4 --master-addr "192.168.1.100"

# 机器2运行  
python train.py --rank 2 --world-size 4 --master-addr "192.168.1.100"
```

## 4. 混合精度训练

### 4.1 自动混合精度 (AMP)

可以节省约50%显存，提升20-50%训练速度：

```python
from torch.cuda.amp import autocast, GradScaler

# 创建梯度缩放器
scaler = GradScaler()

for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)
    
    optimizer.zero_grad()
    
    # 使用autocast进行前向传播
    with autocast():
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
    
    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4.2 混合精度 + 分布式训练

```python
def train_mixed_precision_distributed(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    model = MLPRegressor(...).to(device)
    model = DDP(model, device_ids=[rank])
    
    scaler = GradScaler()
    
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    cleanup()
```

## 5. 模型并行

### 5.1 简单模型并行

适用于模型太大无法放入单个GPU的情况：

```python
class ModelParallelMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        # 前半部分网络在GPU 0
        self.layer1 = nn.Linear(input_size, hidden_sizes[0]).to('cuda:0')
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to('cuda:0')
        
        # 后半部分网络在GPU 1
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2]).to('cuda:1')
        self.layer4 = nn.Linear(hidden_sizes[2], output_size).to('cuda:1')
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 在GPU 0上计算
        x = x.to('cuda:0')
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        
        # 移动到GPU 1继续计算
        x = x.to('cuda:1')
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        
        return x
```

### 5.2 流水线并行

```python
import torch.distributed.pipeline.sync as pipe

# 将模型分割为多个阶段
class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return x

class Stage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# 创建流水线
stages = [Stage1().to('cuda:0'), Stage2().to('cuda:1')]
model = pipe.Pipe(torch.nn.Sequential(*stages), balance=[1, 1], chunks=4)
```

## 6. 性能优化技巧

### 6.1 内存优化

```python
# 1. 梯度累积（减少显存使用）
accumulation_steps = 4
optimizer.zero_grad()

for i, (batch_x, batch_y) in enumerate(train_loader):
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)
    
    with autocast():
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

# 2. 清理显存
torch.cuda.empty_cache()

# 3. 监控显存使用
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"GPU显存 - 已分配: {allocated:.2f}GB, 已预留: {reserved:.2f}GB")
```

### 6.2 计算优化

```python
# 1. 使用更高效的数据类型
model = model.half()  # 转换为半精度

# 2. 启用cuDNN benchmark
torch.backends.cudnn.benchmark = True

# 3. 使用JIT编译
model = torch.jit.script(model)

# 4. 禁用梯度计算（推理时）
with torch.no_grad():
    outputs = model(inputs)
```

### 6.3 数据加载优化

```python
# 使用更多workers和pin_memory
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=min(8, mp.cpu_count()),  # 根据CPU核心数调整
    pin_memory=True,
    persistent_workers=True,  # 保持worker进程存活
    prefetch_factor=2         # 预取数据
)
```

## 7. 最佳实践

### 7.1 选择合适的并行策略

```python
def choose_parallel_strategy():
    """根据资源情况选择并行策略"""
    
    if not torch.cuda.is_available():
        return "CPU训练"
    
    gpu_count = torch.cuda.device_count()
    
    if gpu_count == 1:
        return "单GPU训练"
    elif gpu_count <= 4:
        return "DataParallel或DistributedDataParallel"
    else:
        return "DistributedDataParallel（推荐）"

# 根据GPU数量自动选择策略
strategy = choose_parallel_strategy()
print(f"推荐使用: {strategy}")
```

### 7.2 性能调优检查清单

```python
# ✅ GPU训练优化检查清单

def optimization_checklist():
    checks = {
        "GPU可用性": torch.cuda.is_available(),
        "多GPU": torch.cuda.device_count() > 1,
        "混合精度支持": hasattr(torch.cuda, 'amp'),
        "cuDNN优化": torch.backends.cudnn.enabled,
        "Benchmark模式": torch.backends.cudnn.benchmark,
    }
    
    print("🔍 优化配置检查:")
    for item, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {item}: {status}")
    
    # 显存信息
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory = props.total_memory / (1024**3)
            print(f"  📊 GPU {i}: {props.name} ({memory:.1f}GB)")

# 运行检查
optimization_checklist()
```

### 7.3 训练监控

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """训练时间监控"""
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.2f}秒")

# 使用示例
with timer("整个训练过程"):
    for epoch in range(epochs):
        with timer(f"Epoch {epoch+1}"):
            train_epoch(model, train_loader, optimizer, criterion)
```

## 8. 故障排除

### 8.1 常见错误及解决方案

```python
# 错误1: CUDA out of memory
# 解决方案：
def handle_oom():
    try:
        # 训练代码
        pass
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("显存不足，建议:")
            print("1. 减小batch_size")
            print("2. 使用gradient_accumulation")
            print("3. 启用混合精度训练")
            print("4. 减小模型大小")
            torch.cuda.empty_cache()

# 错误2: 分布式训练挂起
# 确保所有进程同步
dist.barrier()  # 等待所有进程

# 错误3: 数据不一致
# 确保随机种子同步
def set_seed(rank, seed=42):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
```

### 8.2 性能分析工具

```python
# 使用PyTorch Profiler分析性能
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_training"):
        train_epoch(model, train_loader, optimizer, criterion)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 9. 完整示例代码

本项目中提供了完整的实现示例：

- `mlp_explorer.py`: 基础GPU训练支持
- `parallel_training.py`: 完整的并行训练实现
- `gpu_training_examples.py`: GPU训练示例和最佳实践

通过这些文件，你可以学习到：
- 从单GPU到多GPU的完整训练流程
- 不同并行策略的实际应用
- 性能优化和故障排除技巧

## 🎯 总结

PyTorch提供了丰富的并行训练选项：

1. **单GPU**: 最简单的加速方案
2. **DataParallel**: 易于使用的多GPU方案
3. **DistributedDataParallel**: 高效的分布式训练
4. **混合精度**: 显存和速度优化
5. **模型并行**: 处理超大模型

选择合适的方案需要考虑：
- 硬件资源（GPU数量、显存大小）
- 模型大小和复杂度
- 数据集大小
- 训练时间要求

通过本项目的实践，你将掌握这些技术的实际应用! 🚀
