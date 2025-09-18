"""
MLP探索器 - 用于学习多层感知机的交互式工具
通过探究两组数据之间的关系来学习神经网络配置和调参
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class MLPRegressor(nn.Module):
    """可配置的多层感知机回归器"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int = 1,
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ):
        super(MLPRegressor, self).__init__()

        # 激活函数映射
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.01),
            "elu": nn.ELU(),
        }

        self.activation_fn = activation_map.get(activation, nn.ReLU())

        # 构建网络层
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)


class DataGenerator:
    """数据生成器 - 生成不同类型的数据关系"""

    @staticmethod
    def generate_linear_data(
        n_samples: int = 1000, noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成线性关系数据"""
        np.random.seed(42)
        x = np.random.uniform(-2, 2, (n_samples, 1))
        y = 3 * x.flatten() + 2 + np.random.normal(0, noise_level, n_samples)
        return x, y

    @staticmethod
    def generate_quadratic_data(
        n_samples: int = 1000, noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成二次关系数据"""
        np.random.seed(42)
        x = np.random.uniform(-2, 2, (n_samples, 1))
        y = (
            2 * x.flatten() ** 2
            - 3 * x.flatten()
            + 1
            + np.random.normal(0, noise_level, n_samples)
        )
        return x, y

    @staticmethod
    def generate_sine_data(
        n_samples: int = 1000, noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成正弦关系数据"""
        np.random.seed(42)
        x = np.random.uniform(-np.pi, np.pi, (n_samples, 1))
        y = (
            np.sin(x.flatten())
            + 0.5 * np.cos(2 * x.flatten())
            + np.random.normal(0, noise_level, n_samples)
        )
        return x, y

    @staticmethod
    def generate_complex_data(
        n_samples: int = 1000, noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成复杂非线性关系数据"""
        np.random.seed(42)
        x = np.random.uniform(-2, 2, (n_samples, 1))
        y = (
            np.exp(-x.flatten() ** 2)
            + 0.5 * x.flatten() ** 3
            - 0.2 * np.sin(5 * x.flatten())
            + np.random.normal(0, noise_level, n_samples)
        )
        return x, y


class MLPTrainer:
    """MLP训练器"""

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
            self.model = self.model.to(self.device)
            print(f"🔥 使用GPU训练: {torch.cuda.get_device_name()}")

            # 启用数据并行（如果有多个GPU）
            if torch.cuda.device_count() > 1:
                print(f"🚀 启用数据并行，使用 {torch.cuda.device_count()} 个GPU")
                self.model = nn.DataParallel(self.model)
        else:
            self.device = torch.device("cpu")
            print("💻 使用CPU训练")

        # 优化器选择
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []

    def train_epoch(
        self, train_loader: DataLoader, val_loader: DataLoader = None
    ) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            # 将数据移到正确的设备
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs.squeeze(), batch_y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
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

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """完整训练过程"""
        self.train_losses = []
        self.val_losses = []

        for epoch in range(epochs):
            train_loss, val_loss = self.train_epoch(train_loader, val_loader)
            self.train_losses.append(train_loss)
            if val_loader is not None:
                self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 20 == 0:
                if val_loader is not None:
                    print(
                        f"Epoch [{epoch+1}/{epochs}], 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}"
                    )
                else:
                    print(f"Epoch [{epoch+1}/{epochs}], 训练损失: {train_loss:.4f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses if val_loader is not None else [],
        }


class MLPVisualizer:
    """MLP可视化器"""

    def __init__(self):
        self.fig_size = (15, 10)

    def plot_data_relationship(
        self, x: np.ndarray, y: np.ndarray, title: str = "数据关系"
    ):
        """绘制数据关系图"""
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.6, s=20)
        plt.xlabel("输入特征 (X)")
        plt.ylabel("目标变量 (Y)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_training_history(
        self, train_losses: List[float], val_losses: List[float] = None
    ):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 1, 1)
        plt.plot(train_losses, label="训练损失", linewidth=2)
        if val_losses:
            plt.plot(val_losses, label="验证损失", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("损失")
        plt.title("训练过程中的损失变化")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        plt.tight_layout()
        plt.show()

    def plot_predictions(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        title: str = "预测结果对比",
    ):
        """绘制预测结果"""
        plt.figure(figsize=self.fig_size)

        # 排序以便更好地绘制预测线
        sort_idx = np.argsort(x_test.flatten())
        x_sorted = x_test.flatten()[sort_idx]
        y_test_sorted = y_test[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        plt.subplot(2, 2, 1)
        plt.scatter(x_test, y_test, alpha=0.6, label="真实值", s=20)
        plt.plot(x_sorted, y_pred_sorted, "r-", label="预测值", linewidth=2)
        plt.xlabel("输入特征 (X)")
        plt.ylabel("目标变量 (Y)")
        plt.title("预测 vs 真实值")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.6, s=20)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            linewidth=2,
        )
        plt.xlabel("真实值")
        plt.ylabel("预测值")
        plt.title("预测值 vs 真实值散点图")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
        plt.xlabel("预测值")
        plt.ylabel("残差")
        plt.title("残差图")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
        plt.xlabel("残差")
        plt.ylabel("频数")
        plt.title("残差分布")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 计算并显示评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\n评估指标:")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"决定系数 (R²): {r2:.4f}")


class MLPExplorer:
    """MLP探索器主类"""

    def __init__(self):
        self.visualizer = MLPVisualizer()
        self.scaler = StandardScaler()

    def explore_mlp(
        self,
        data_type: str = "quadratic",
        hidden_sizes: List[int] = [64, 32],
        learning_rate: float = 0.001,
        epochs: int = 200,
        batch_size: int = 32,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        optimizer_type: str = "adam",
        noise_level: float = 0.1,
        n_samples: int = 1000,
        use_gpu: bool = True,
    ):
        """
        完整的MLP探索流程

        参数:
        - data_type: 数据类型 ['linear', 'quadratic', 'sine', 'complex']
        - hidden_sizes: 隐藏层大小列表，如 [64, 32] 表示两个隐藏层
        - learning_rate: 学习率
        - epochs: 训练轮数
        - batch_size: 批次大小
        - activation: 激活函数 ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
        - dropout_rate: Dropout比率 (0.0-1.0)
        - optimizer_type: 优化器类型 ['adam', 'sgd', 'rmsprop']
        - noise_level: 噪声水平
        - n_samples: 样本数量
        - use_gpu: 是否使用GPU加速训练（支持多GPU数据并行）
        """

        print(f"🚀 开始MLP探索...")
        print(f"数据类型: {data_type}")
        print(f"网络结构: 输入层(1) -> 隐藏层{hidden_sizes} -> 输出层(1)")
        print(
            f"训练参数: 学习率={learning_rate}, epochs={epochs}, batch_size={batch_size}"
        )
        print(
            f"网络参数: 激活函数={activation}, dropout={dropout_rate}, 优化器={optimizer_type}"
        )
        print("-" * 60)

        # 1. 生成数据
        data_generators = {
            "linear": DataGenerator.generate_linear_data,
            "quadratic": DataGenerator.generate_quadratic_data,
            "sine": DataGenerator.generate_sine_data,
            "complex": DataGenerator.generate_complex_data,
        }

        if data_type not in data_generators:
            raise ValueError(f"不支持的数据类型: {data_type}")

        x, y = data_generators[data_type](n_samples, noise_level)

        # 显示原始数据
        print("📊 生成的数据关系:")
        self.visualizer.plot_data_relationship(
            x, y, f"{data_type.capitalize()} 数据关系"
        )

        # 2. 数据预处理
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 3. 创建模型
        model = MLPRegressor(
            input_size=1,
            hidden_sizes=hidden_sizes,
            output_size=1,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        print(f"\n🧠 模型架构:")
        print(model)

        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params}")
        print(f"可训练参数数量: {trainable_params}")

        # 4. 训练模型
        print(f"\n🔥 开始训练...")
        trainer = MLPTrainer(model, learning_rate, optimizer_type, use_gpu)
        history = trainer.train(train_loader, val_loader, epochs, verbose=True)

        # 5. 可视化训练过程
        print(f"\n📈 训练历史可视化:")
        self.visualizer.plot_training_history(
            history["train_losses"], history["val_losses"]
        )

        # 6. 模型预测和评估
        model.eval()
        with torch.no_grad():
            # 将测试数据移到模型所在的设备
            x_test_device = x_test_tensor.to(trainer.device)
            y_pred_tensor = model(x_test_device)
            y_pred = y_pred_tensor.cpu().squeeze().numpy()

        # 7. 结果可视化
        print(f"\n🎯 预测结果可视化:")
        x_test_original = self.scaler.inverse_transform(x_test)
        self.visualizer.plot_predictions(
            x_test_original,
            y_test,
            y_pred,
            f"{data_type.capitalize()} 数据 - MLP预测结果",
        )

        return {
            "model": model,
            "history": history,
            "predictions": y_pred,
            "test_data": (x_test_original, y_test),
            "metrics": {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred),
            },
        }


if __name__ == "__main__":
    # 创建MLP探索器实例
    explorer = MLPExplorer()

    print("=" * 80)
    print("🎓 MLP学习探索器")
    print("探究两组数据之间的关系，学习神经网络配置和调参")
    print("=" * 80)

    # 示例1: 二次关系数据的基础探索
    print("\n📚 示例1: 二次关系数据 - 基础配置")
    result1 = explorer.explore_mlp(
        data_type="quadratic",
        hidden_sizes=[32, 16],
        learning_rate=0.01,
        epochs=150,
        batch_size=32,
        activation="relu",
    )
