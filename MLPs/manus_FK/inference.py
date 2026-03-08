# infer_fk.py
import torch, numpy as np
import json
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trainer import FKMLP
from dataset import transform_to_local_coordinates
import os

# 加载模型
ckpt = torch.load(
    os.path.join(os.path.dirname(__file__), "checkpoint", "fk_mlp_best.pth"),
    map_location="cpu",
    weights_only=False,  # 设置为False以兼容旧版本的checkpoint
)
dof, out_dim = ckpt["dof"], ckpt["out_dim"]

# 检查是否使用不确定性加权
heteroscedastic = ckpt.get("heteroscedastic", False)
print(f"模型配置: DOF={dof}, out_dim={out_dim}, heteroscedastic={heteroscedastic}")

model = FKMLP(dof, out_dim, heteroscedastic=heteroscedastic)
model.load_state_dict(ckpt["model"])
model.eval()

x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32)
x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32)
y_mean = torch.tensor(ckpt["y_mean"], dtype=torch.float32)
y_std = torch.tensor(ckpt["y_std"], dtype=torch.float32)


def predict_xyz(angles_np):  # angles_np: [DOF] 原始角度
    x = torch.tensor(angles_np, dtype=torch.float32).unsqueeze(0)  # [1, DOF]
    x_n = (x - x_mean) / x_std
    with torch.no_grad():
        y_n = model(x_n)  # [1, out_dim] 标准化空间

    # 检查是否是heteroscedastic输出
    if heteroscedastic and y_n.size(1) == out_dim * 2:  # heteroscedastic: 6P
        mu_n, s_n = torch.chunk(y_n, 2, dim=-1)  # 分离均值和方差
        y = mu_n * y_std + y_mean  # 只使用均值进行预测
    else:  # 传统输出: 3P
        y = y_n * y_std + y_mean  # 反标准化到真实 xyz

    xyz = y.view(-1, 3).numpy()  # [num_links, 3]
    return xyz


def load_random_samples(jsonl_path, num_samples=5):
    """从jsonl文件中随机抽取指定数量的样本"""
    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    # 随机选择5行
    selected_lines = random.sample(lines, min(num_samples, len(lines)))

    samples = []
    for line in selected_lines:
        data = json.loads(line)
        if "left" in data and "angles" in data["left"] and "poses" in data["left"]:
            samples.append(
                {
                    "angles": np.array(data["left"]["angles"], dtype=np.float32),
                    "poses": data["left"]["poses"],
                }
            )

    return samples


def extract_tip_positions(poses, use_local_coordinates=True):
    """从poses中提取每个手指的tip位置
    如果use_local_coordinates=True，会先将世界坐标转换为局部坐标
    """
    # 如果需要使用局部坐标系，先进行坐标变换
    if use_local_coordinates:
        poses = transform_to_local_coordinates(poses)

    # 根据数据结构，每个pose是[x, y, z, qw, qx, qy, qz]
    # 我们取前3个元素作为xyz坐标
    tip_positions = []
    for pose in poses:
        if len(pose) >= 3:
            tip_positions.append([pose[0], pose[1], pose[2]])
    return np.array(tip_positions)


def calculate_3d_error(pred_positions, true_positions):
    """计算3D位置误差"""
    # 确保两个数组形状相同
    min_len = min(len(pred_positions), len(true_positions))
    pred_positions = pred_positions[:min_len]
    true_positions = true_positions[:min_len]

    # 计算每个点的3D距离
    errors = np.linalg.norm(pred_positions - true_positions, axis=1)
    return errors


def visualize_hand_pose(
    true_poses, pred_poses, title="Hand Pose", ax=None, use_local_coordinates=True
):
    """可视化手部姿势，显示真实值（红色）和预测值（蓝色）"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # 如果需要使用局部坐标系，先对真实值进行坐标变换
    if use_local_coordinates:
        true_poses = transform_to_local_coordinates(true_poses)

    # 提取真实关节的xyz坐标
    true_x = [pose[0] for pose in true_poses]
    true_y = [pose[1] for pose in true_poses]
    true_z = [pose[2] for pose in true_poses]

    # 提取预测关节的xyz坐标
    pred_x = [pose[0] for pose in pred_poses]
    pred_y = [pose[1] for pose in pred_poses]
    pred_z = [pose[2] for pose in pred_poses]

    # 绘制真实关节点（红色）
    ax.scatter(true_x, true_y, true_z, c="red", s=50, alpha=0.8, label="Ground Truth")

    # 绘制预测关节点（蓝色）
    ax.scatter(pred_x, pred_y, pred_z, c="blue", s=50, alpha=0.8, label="Prediction")

    # 为每个点添加索引标签
    for i in range(len(true_poses)):
        ax.text(true_x[i], true_y[i], true_z[i], f" {i}", fontsize=8, color="red")
        ax.text(pred_x[i], pred_y[i], pred_z[i], f" {i}", fontsize=8, color="blue")

    # 定义手指连接关系
    # 0为原点，1-4为thumb，5-9为index，10-14为middle，15-19为ring，20-24为pinky
    finger_connections = [
        # 拇指 (thumb): 0-1-2-3-4
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        # 食指 (index): 0-5-6-7-8-9
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        # 中指 (middle): 0-10-11-12-13-14
        [0, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        # 无名指 (ring): 0-15-16-17-18-19
        [0, 15],
        [15, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        # 小指 (pinky): 0-20-21-22-23-24
        [0, 20],
        [20, 21],
        [21, 22],
        [22, 23],
        [23, 24],
    ]

    # 绘制真实值的连线（红色虚线）
    for connection in finger_connections:
        if connection[0] < len(true_poses) and connection[1] < len(true_poses):
            ax.plot(
                [true_x[connection[0]], true_x[connection[1]]],
                [true_y[connection[0]], true_y[connection[1]]],
                [true_z[connection[0]], true_z[connection[1]]],
                "r--",
                alpha=0.6,
                linewidth=1,
            )

    # 绘制预测值的连线（蓝色实线）
    for connection in finger_connections:
        if connection[0] < len(pred_poses) and connection[1] < len(pred_poses):
            ax.plot(
                [pred_x[connection[0]], pred_x[connection[1]]],
                [pred_y[connection[0]], pred_y[connection[1]]],
                [pred_z[connection[0]], pred_z[connection[1]]],
                "b-",
                alpha=0.6,
                linewidth=1,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    return ax


# 主程序
def main():
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)

    # 加载数据文件路径
    jsonl_path = os.path.join(os.path.dirname(__file__), "ergo_data_0.jsonl")

    # 从数据中随机抽取5个样本
    print(f"正在从{jsonl_path}中随机抽取5个样本...")
    samples = load_random_samples(jsonl_path, 5)
    print(f"成功加载了 {len(samples)} 个样本")

    # 存储所有误差用于统计
    all_errors = []

    # 创建可视化图形
    fig = plt.figure(figsize=(20, 12))

    # 处理每个样本
    for i, sample in enumerate(samples):
        print(f"\n处理样本 {i+1}:")

        # 获取真实角度和pose数据
        angles = sample["angles"]
        true_poses = sample["poses"]

        # 提取真实的tip位置（应用坐标变换到局部坐标系）
        # 注意：模型训练时使用的是局部坐标系，所以这里也需要将ground truth转换到局部坐标系
        true_tip_positions = extract_tip_positions(
            true_poses, use_local_coordinates=True
        )
        print(f"  真实tip位置数量: {len(true_tip_positions)}")

        # 使用模型预测
        pred_xyz = predict_xyz(angles)
        print(f"  预测位置数量: {len(pred_xyz)}")

        # 计算误差
        errors = calculate_3d_error(pred_xyz, true_tip_positions)
        mean_error = np.mean(errors)
        all_errors.extend(errors)

        print(f"  平均3D误差: {mean_error:.4f} 米")
        print(f"  最大误差: {np.max(errors):.4f} 米")
        print(f"  最小误差: {np.min(errors):.4f} 米")

        # 可视化（注意：现在显示的是局部坐标系下的位置）
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        visualize_hand_pose(
            true_poses,
            pred_xyz,
            f"Sample {i+1} - Local Coordinates",
            ax,
            use_local_coordinates=True,
        )

    # 添加总体统计图
    ax_stats = fig.add_subplot(2, 3, 6, projection="3d")
    ax_stats.text(
        0.1,
        0.5,
        0.5,
        f"Overall Statistics:\nMean Error: {np.mean(all_errors):.4f}m\nMax Error: {np.max(all_errors):.4f}m\nMin Error: {np.min(all_errors):.4f}m\nTotal Points: {len(all_errors)}",
        transform=ax_stats.transAxes,
        fontsize=12,
        verticalalignment="center",
    )
    ax_stats.set_title("Error Statistics")
    ax_stats.axis("off")

    plt.tight_layout()
    plt.show()

    # 打印总体统计
    print(f"\n=== 总体统计 ===")
    print(f"所有样本的平均3D误差: {np.mean(all_errors):.4f} 米")
    print(f"最大误差: {np.max(all_errors):.4f} 米")
    print(f"最小误差: {np.min(all_errors):.4f} 米")
    print(f"标准差: {np.std(all_errors):.4f} 米")
    print(f"总样本数: {len(all_errors)}")


if __name__ == "__main__":
    main()
