"""
MLP并行训练实现
支持数据并行(DataParallel)和分布式数据并行(DistributedDataParallel)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import time
from typing import Tuple, List, Dict
import numpy as np
from mlp_explorer import MLPRegressor, DataGenerator, MLPVisualizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class ParallelMLPTrainer:
    """支持并行训练的MLP训练器"""

    def __init__(
        self,
        model: MLPRegressor,
        learning_rate: float = 0.001,
        optimizer_type: str = "adam",
        use_gpu: bool = True,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # 设置设备
        if self.use_gpu:
            self.device = torch.device("cuda")
            print(f"🔥 使用GPU训练: {torch.cuda.get_device_name()}")
            print(f"📊 可用GPU数量: {torch.cuda.device_count()}")
        else:
            self.device = torch.device("cpu")
            print("💻 使用CPU训练")

        # 将模型移到设备上
        self.model = self.model.to(self.device)

        # 设置优化器
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []

    def setup_data_parallel(self):
        """设置数据并行训练"""
        if torch.cuda.device_count() > 1:
            print(f"🚀 启用数据并行训练，使用 {torch.cuda.device_count()} 个GPU")
            self.model = nn.DataParallel(self.model)
            return True
        else:
            print("⚠️ 只有一个GPU可用，无法使用数据并行")
            return False

    def train_epoch_parallel(
        self, train_loader: DataLoader, val_loader: DataLoader = None
    ) -> Tuple[float, float]:
        """并行训练一个epoch"""
        self.model.train()
        train_loss = 0.0
        num_batches = len(train_loader)

        # 训练阶段
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # 将数据移到GPU
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs.squeeze(), batch_y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # 显示进度
            if batch_idx % (num_batches // 4) == 0:
                print(f"  批次 [{batch_idx}/{num_batches}], 损失: {loss.item():.4f}")

        train_loss /= len(train_loader)

        # 验证阶段
        val_loss = 0.0
        if val_loader is not None:
            self.model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

        return train_loss, val_loss

    def train_parallel(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        verbose: bool = True,
        use_data_parallel: bool = True,
    ) -> Dict:
        """并行训练主函数"""

        # 设置数据并行
        is_parallel = False
        if use_data_parallel and self.use_gpu:
            is_parallel = self.setup_data_parallel()

        self.train_losses = []
        self.val_losses = []

        # 记录训练开始时间
        start_time = time.time()

        print(f"\n🎯 开始{'并行' if is_parallel else '单卡'}训练...")
        print(f"📈 总epochs: {epochs}")
        print("-" * 50)

        for epoch in range(epochs):
            epoch_start_time = time.time()

            train_loss, val_loss = self.train_epoch_parallel(train_loader, val_loader)
            self.train_losses.append(train_loss)
            if val_loader is not None:
                self.val_losses.append(val_loss)

            epoch_time = time.time() - epoch_start_time

            if verbose and (epoch + 1) % 20 == 0:
                if val_loader is not None:
                    print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.2f}s)")
                    print(f"  训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.2f}s)")
                    print(f"  训练损失: {train_loss:.4f}")

        total_time = time.time() - start_time
        print(f"\n⏱️ 训练完成! 总耗时: {total_time:.2f}秒")
        print(f"📊 平均每epoch: {total_time/epochs:.2f}秒")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses if val_loader is not None else [],
            "training_time": total_time,
            "epochs_per_second": epochs / total_time,
        }


def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def train_distributed(rank, world_size, model_config, data_config, train_config):
    """分布式训练函数"""
    # 设置分布式环境
    setup_distributed(rank, world_size)

    # 设置设备
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 创建模型并移到对应GPU
    model = MLPRegressor(**model_config).to(device)

    # 包装为DDP模型
    model = DDP(model, device_ids=[rank])

    # 生成数据
    x, y = DataGenerator.generate_quadratic_data(**data_config)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42
    )

    # 创建数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )

    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        sampler=train_sampler,
        pin_memory=True,
    )

    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    criterion = nn.MSELoss()

    # 训练循环
    model.train()
    for epoch in range(train_config["epochs"]):
        train_sampler.set_epoch(epoch)  # 重要：设置epoch以确保数据打乱

        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if rank == 0 and (epoch + 1) % 20 == 0:  # 只在主进程打印
            print(
                f'Epoch [{epoch+1}/{train_config["epochs"]}], 损失: {epoch_loss/len(train_loader):.4f}'
            )

    # 清理
    cleanup_distributed()


class ParallelMLPExplorer:
    """支持并行训练的MLP探索器"""

    def __init__(self):
        self.visualizer = MLPVisualizer()
        self.scaler = StandardScaler()

    def compare_training_methods(
        self,
        data_type: str = "quadratic",
        hidden_sizes: List[int] = [128, 64, 32],
        epochs: int = 100,
        batch_size: int = 64,
        n_samples: int = 10000,
    ):
        """比较不同训练方法的性能"""

        print("🔬 并行训练方法对比实验")
        print("=" * 60)

        # 生成数据
        data_generators = {
            "linear": DataGenerator.generate_linear_data,
            "quadratic": DataGenerator.generate_quadratic_data,
            "sine": DataGenerator.generate_sine_data,
            "complex": DataGenerator.generate_complex_data,
        }

        x, y = data_generators[data_type](n_samples, 0.1)
        x_scaled = self.scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y, test_size=0.2, random_state=42
        )

        # 转换为PyTorch张量
        x_train_tensor = torch.FloatTensor(x_train)
        y_train_tensor = torch.FloatTensor(y_train)
        x_test_tensor = torch.FloatTensor(x_test)
        y_test_tensor = torch.FloatTensor(y_test)

        # 创建数据加载器
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        val_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        results = {}

        # 1. CPU训练
        print("\n💻 方法1: CPU训练")
        model_cpu = MLPRegressor(input_size=1, hidden_sizes=hidden_sizes, output_size=1)
        trainer_cpu = ParallelMLPTrainer(model_cpu, learning_rate=0.001, use_gpu=False)
        result_cpu = trainer_cpu.train_parallel(
            train_loader, val_loader, epochs, verbose=False, use_data_parallel=False
        )
        results["CPU"] = result_cpu

        # 2. 单GPU训练
        if torch.cuda.is_available():
            print("\n🔥 方法2: 单GPU训练")
            model_gpu = MLPRegressor(
                input_size=1, hidden_sizes=hidden_sizes, output_size=1
            )
            trainer_gpu = ParallelMLPTrainer(
                model_gpu, learning_rate=0.001, use_gpu=True
            )
            result_gpu = trainer_gpu.train_parallel(
                train_loader, val_loader, epochs, verbose=False, use_data_parallel=False
            )
            results["单GPU"] = result_gpu

            # 3. 数据并行训练（如果有多个GPU）
            if torch.cuda.device_count() > 1:
                print("\n🚀 方法3: 数据并行训练")
                model_parallel = MLPRegressor(
                    input_size=1, hidden_sizes=hidden_sizes, output_size=1
                )
                trainer_parallel = ParallelMLPTrainer(
                    model_parallel, learning_rate=0.001, use_gpu=True
                )
                result_parallel = trainer_parallel.train_parallel(
                    train_loader,
                    val_loader,
                    epochs,
                    verbose=False,
                    use_data_parallel=True,
                )
                results["数据并行"] = result_parallel

        # 显示结果对比
        self.display_comparison_results(results)

        return results

    def display_comparison_results(self, results: Dict):
        """显示训练方法对比结果"""
        print("\n📊 训练方法性能对比")
        print("=" * 80)
        print(
            f"{'方法':<12} | {'训练时间(秒)':<12} | {'训练速度(epochs/s)':<16} | {'最终损失':<10}"
        )
        print("-" * 80)

        for method, result in results.items():
            training_time = result["training_time"]
            speed = result["epochs_per_second"]
            final_loss = result["train_losses"][-1] if result["train_losses"] else 0

            print(
                f"{method:<12} | {training_time:<12.2f} | {speed:<16.2f} | {final_loss:<10.4f}"
            )

        # 计算加速比
        if "CPU" in results and len(results) > 1:
            cpu_time = results["CPU"]["training_time"]
            print(f"\n🚀 加速比对比 (相对于CPU):")
            print("-" * 40)
            for method, result in results.items():
                if method != "CPU":
                    speedup = cpu_time / result["training_time"]
                    print(f"{method:<12}: {speedup:.2f}x 加速")

    def demonstrate_mixed_precision_training(
        self,
        data_type: str = "complex",
        hidden_sizes: List[int] = [256, 128, 64, 32],
        epochs: int = 100,
    ):
        """演示混合精度训练"""

        print("\n⚡ 混合精度训练演示")
        print("=" * 50)

        if not torch.cuda.is_available():
            print("⚠️ 需要GPU支持混合精度训练")
            return

        # 生成数据
        x, y = DataGenerator.generate_complex_data(5000, 0.1)
        x_scaled = self.scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y, test_size=0.2, random_state=42
        )

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(x_train), torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, pin_memory=True
        )

        # 创建模型
        model = MLPRegressor(
            input_size=1, hidden_sizes=hidden_sizes, output_size=1
        ).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 创建GradScaler用于混合精度训练
        scaler = torch.cuda.amp.GradScaler()

        print("🔥 开始混合精度训练...")
        start_time = time.time()

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.cuda(non_blocking=True)
                batch_y = batch_y.cuda(non_blocking=True)

                optimizer.zero_grad()

                # 使用autocast进行前向传播
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)

                # 缩放损失并反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], 损失: {epoch_loss/len(train_loader):.4f}"
                )

        training_time = time.time() - start_time
        print(f"\n⏱️ 混合精度训练完成! 耗时: {training_time:.2f}秒")
        print(f"🎯 内存使用更少，训练速度更快！")


def run_distributed_training_example():
    """运行分布式训练示例"""
    print("🌐 分布式训练示例")
    print("=" * 50)

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("⚠️ 分布式训练需要至少2个GPU")
        return

    # 配置参数
    model_config = {
        "input_size": 1,
        "hidden_sizes": [128, 64, 32],
        "output_size": 1,
        "activation": "relu",
    }

    data_config = {"n_samples": 5000, "noise_level": 0.1}

    train_config = {"batch_size": 64, "learning_rate": 0.001, "epochs": 100}

    world_size = torch.cuda.device_count()
    print(f"🚀 启动 {world_size} 个进程进行分布式训练...")

    # 启动多进程分布式训练
    mp.spawn(
        train_distributed,
        args=(world_size, model_config, data_config, train_config),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    print("🎓 MLP并行训练教程")
    print("=" * 80)

    # 创建并行探索器
    explorer = ParallelMLPExplorer()

    # 运行训练方法对比
    print("\n1️⃣ 运行训练方法对比实验...")
    comparison_results = explorer.compare_training_methods(
        data_type="quadratic",
        hidden_sizes=[128, 64, 32],
        epochs=50,  # 减少epochs以加快演示
        n_samples=5000,
    )

    # 演示混合精度训练
    print("\n\n2️⃣ 演示混合精度训练...")
    explorer.demonstrate_mixed_precision_training(epochs=50)  # 减少epochs以加快演示

    # 分布式训练示例（需要多GPU）
    print("\n\n3️⃣ 分布式训练示例...")
    run_distributed_training_example()

    print("\n🎉 并行训练教程完成!")
    print("💡 主要学习点:")
    print("  - 数据并行可以有效利用多GPU")
    print("  - 混合精度训练节省显存并加速")
    print("  - 分布式训练适合大规模数据")
    print("  - 选择合适的并行策略很重要")
