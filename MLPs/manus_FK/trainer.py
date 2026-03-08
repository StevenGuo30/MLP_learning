# train_fk.py
import torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split
from dataset import ErgoFKDataset
import os

# 假设你有一个手的拓扑结构 (要根据你的数据顺序来定)
# 示例：[(0,1), (1,2), (2,3), ...]
edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # 拇指
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # 食指
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # 中指
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # 无名指
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # 小指
    (0, 21),
    (21, 22),
    (22, 23),
    (23, 24),
]  # 掌心/其余


class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        h = torch.relu(self.fc1(self.ln(x)))
        h = self.fc2(h)
        return x + h


class FKMLP(nn.Module):
    def __init__(
        self,
        dof,
        out_dim,
        hidden=(256, 256, 256),
        use_finger_heads=True,
        heteroscedastic=True,
    ):
        super().__init__()
        self.use_finger_heads = use_finger_heads
        self.heteroscedastic = heteroscedastic
        self.num_joints = out_dim // 3  # 25个关节

        # 输出维度：heteroscedastic时输出6P（μ+s），否则输出3P
        self.output_dim = out_dim * 2 if heteroscedastic else out_dim

        # 公共特征提取主干
        layers = []
        in_dim = dof
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h

        self.backbone = nn.Sequential(*layers)

        if use_finger_heads:
            # 为每根手指创建独立的预测头
            # 拇指(4个关节), 食指(5个关节), 中指(5个关节), 无名指(5个关节), 小指(5个关节), 掌心(1个关节)
            finger_sizes = [4, 5, 5, 5, 5, 1]  # 每根手指的关节数
            self.finger_heads = nn.ModuleList()

            for finger_size in finger_sizes:
                # 每个手指输出：μ(3*finger_size) + s(3*finger_size) = 6*finger_size
                head_output_dim = (
                    finger_size * 6 if heteroscedastic else finger_size * 3
                )
                head = nn.Sequential(
                    ResBlock(in_dim),
                    ResBlock(in_dim),
                    nn.Linear(in_dim, head_output_dim),
                )
                self.finger_heads.append(head)
        else:
            # 传统全连接输出
            self.output_head = nn.Sequential(
                ResBlock(in_dim), ResBlock(in_dim), nn.Linear(in_dim, self.output_dim)
            )

    def forward(self, q):  # q: [B, DOF] (已标准化)
        features = self.backbone(q)  # [B, hidden_dim]

        if self.use_finger_heads:
            # 使用多头输出
            outputs = []
            for head in self.finger_heads:
                outputs.append(head(features))
            return torch.cat(outputs, dim=1)  # [B, output_dim]
        else:
            # 传统输出
            return self.output_head(features)  # [B, output_dim]


# ===== helpers: 从 DataLoader 里拿到底层 dataset 的均值方差 =====
def _get_base_dataset(loader):
    # DataLoader -> Subset(train/val) -> ErgoFKDataset
    base = getattr(loader.dataset, "dataset", loader.dataset)
    return base


def compute_physical_metrics(model, loader, device, exclude_root=True):
    ds = _get_base_dataset(loader)
    x_mean = torch.tensor(ds.x_mean, dtype=torch.float32, device=device)
    x_std = torch.tensor(ds.x_std, dtype=torch.float32, device=device)
    y_mean = torch.tensor(ds.y_mean, dtype=torch.float32, device=device)
    y_std = torch.tensor(ds.y_std, dtype=torch.float32, device=device)

    model.eval()
    total_err = 0.0
    total_cnt = 0
    with torch.no_grad():
        for x_n, y_n in loader:
            x_n = x_n.to(device)  # 标准化空间
            y_n = y_n.to(device)

            # 网络输出
            yhat_n = model(x_n)  # [B, 6P] 或 [B, 3P]

            # 检查是否是heteroscedastic输出
            if yhat_n.size(1) == y_n.size(1) * 2:  # heteroscedastic: 6P
                mu_n, s_n = torch.chunk(yhat_n, 2, dim=-1)  # 分离均值和方差
                yhat = mu_n * y_std + y_mean  # 只使用均值进行物理误差计算
            else:  # 传统输出: 3P
                yhat = yhat_n * y_std + y_mean

            y = y_n * y_std + y_mean  # 物理空间

            B = y.size(0)
            P = y.size(1) // 3  # 点的个数(例如 25)

            yhat = yhat.view(B, P, 3)
            y = y.view(B, P, 3)

            if exclude_root and P >= 1:
                yhat = yhat[:, 1:, :]  # 去掉 index=0 (手腕)
                y = y[:, 1:, :]

            # 逐点欧氏距离 -> 平均
            d = torch.linalg.norm(yhat - y, dim=-1)  # [B, P']
            total_err += d.sum().item()
            total_cnt += d.numel()

    mean_err_m = total_err / max(total_cnt, 1)  # 单位: 米 (你的坐标是米)
    mean_err_cm = mean_err_m * 100.0
    return mean_err_m, mean_err_cm


def smooth_l1_loss(pred, target, beta=1.0, reduction="mean"):
    """Smooth L1 (Huber) Loss - 对异常点更鲁棒"""
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def weighted_finger_loss(pred, target, finger_weights=None):
    """多任务损失：全点 + 指尖加权
    指尖关节索引：4, 9, 14, 19, 24 (每根手指的最后一个关节)
    """
    if finger_weights is None:
        # 默认权重：指尖关节权重为2，其他关节权重为1
        finger_weights = torch.ones(25)  # 25个关节
        fingertip_indices = [4, 9, 14, 19, 24]  # 指尖关节索引
        for idx in fingertip_indices:
            finger_weights[idx] = 2.0

    # 将权重移到正确的设备上
    finger_weights = finger_weights.to(pred.device)

    # 计算每个关节的损失 (不进行平均)
    joint_losses = smooth_l1_loss(pred, target, beta=1.0, reduction="none")

    # 应用权重
    B = pred.size(0)
    P = pred.size(1) // 3  # 关节数量
    joint_losses = joint_losses.view(B, P, 3)  # [B, P, 3]

    # 对每个关节的3D位置求平均，然后应用权重
    joint_losses = joint_losses.mean(dim=2)  # [B, P]
    weighted_losses = joint_losses * finger_weights.unsqueeze(0)  # [B, P]

    return weighted_losses.mean()


def heteroscedastic_loss(mu, s, target):
    """不确定性加权损失 (Heteroscedastic Regression)
    mu: [B, 3P] 均值预测
    s: [B, 3P] 对数方差预测
    target: [B, 3P] 真实值
    """
    err2 = (mu - target).pow(2)  # [B, 3P]
    loss = torch.exp(-s) * err2 + s  # exp(-s) * ||y-μ||² + s
    return loss.mean()


def weighted_finger_loss_heteroscedastic(mu, s, target, finger_weights=None):
    """不确定性加权的指尖损失"""
    if finger_weights is None:
        # 默认权重：指尖关节权重为2，其他关节权重为1
        finger_weights = torch.ones(25)  # 25个关节
        fingertip_indices = [4, 9, 14, 19, 24]  # 指尖关节索引
        for idx in fingertip_indices:
            finger_weights[idx] = 2.0

    # 将权重移到正确的设备上
    finger_weights = finger_weights.to(mu.device)

    # 计算不确定性加权损失
    err2 = (mu - target).pow(2)  # [B, 3P]
    loss_per_element = torch.exp(-s) * err2 + s  # [B, 3P]

    # 应用权重
    B = mu.size(0)
    P = mu.size(1) // 3  # 关节数量
    loss_per_element = loss_per_element.view(B, P, 3)  # [B, P, 3]

    # 对每个关节的3D位置求平均，然后应用权重
    loss_per_joint = loss_per_element.mean(dim=2)  # [B, P]
    weighted_losses = loss_per_joint * finger_weights.unsqueeze(0)  # [B, P]

    return weighted_losses.mean()


def bone_length_loss(yhat, y, edges):
    # yhat,y: [B, P*3]，物理空间
    B = y.size(0)
    P = y.size(1) // 3
    yhat = yhat.view(B, P, 3)
    y = y.view(B, P, 3)

    def lengths(z):
        return torch.stack(
            [torch.linalg.norm(z[:, i] - z[:, j], dim=-1) for i, j in edges], dim=-1
        )

    return torch.nn.functional.l1_loss(lengths(yhat), lengths(y))


def main():
    jsonl_path = os.path.join(
        os.path.dirname(__file__), "ergo_data_1.jsonl"
    )  # 改成你的实际路径
    hand = "left"  # or "right"
    ds = ErgoFKDataset(
        jsonl_path, hand=hand, standardize=True, use_local_coordinates=True
    )

    N, dof, out_dim = len(ds), ds.X.shape[1], ds.Y.shape[1]
    print(f"Samples={N}, DOF={dof}, out_dim={out_dim} (num_links={out_dim//3})")

    # 划分训练/验证
    val_ratio = 0.2
    n_val = int(N * val_ratio)
    n_train = N - n_val
    tr_ds, va_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=512, shuffle=False, num_workers=0)

    # 创建loaders字典以便后续使用
    loaders = {"train": tr_loader, "val": va_loader}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建增强的模型（启用不确定性加权）
    model = FKMLP(
        dof=dof,
        out_dim=out_dim,
        hidden=(256, 256, 256),
        use_finger_heads=True,
        heteroscedastic=True,
    )
    model = model.to(device)  # 将模型移到设备上
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 余弦退火调度器 + Warmup
    total_epochs = 2000
    warmup_epochs = 50
    total_steps = total_epochs * len(tr_loader)
    warmup_steps = warmup_epochs * len(tr_loader)

    # 余弦退火调度器（warmup后开始）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=1e-5
    )

    # EMA (Exponential Moving Average)
    ema_decay = 0.999
    ema = {k: p.detach().clone() for k, p in model.state_dict().items()}

    # 使用新的损失函数组合
    # loss_fn = nn.MSELoss()  # 不再使用纯MSE

    # 获取标准化参数用于骨长损失计算
    y_mean = torch.tensor(ds.y_mean, dtype=torch.float32, device=device)
    y_std = torch.tensor(ds.y_std, dtype=torch.float32, device=device)

    best_cm = float("inf")  # 初始化best_cm
    step = 0

    for epoch in range(total_epochs):
        model.train()
        tr_loss = 0.0
        ntr = 0

        for x, y in tr_loader:
            x = x.to(device)
            y = y.to(device)

            # 不确定性加权输出
            yhat_n = model(x)  # [B, 6P] in normalized space
            mu_n, s_n = torch.chunk(yhat_n, 2, dim=-1)  # 分离均值和方差

            # 反标准化到物理空间
            mu = mu_n * y_std + y_mean  # [B, 3P]
            y_phys = y * y_std + y_mean  # [B, 3P]

            # 1. 不确定性加权损失
            loss_hetero = weighted_finger_loss_heteroscedastic(mu_n, s_n, y)

            # 2. 骨长损失（物理空间）
            loss_bone = bone_length_loss(mu, y_phys, edges)

            # 3. 总损失组合
            loss = loss_hetero + 0.1 * loss_bone  # λ=0.1 可调

            opt.zero_grad()
            loss.backward()
            opt.step()

            # 更新EMA
            with torch.no_grad():
                for k, p in model.state_dict().items():
                    ema[k].mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)

            # 学习率调度（Warmup + Cosine）
            if step < warmup_steps:
                # 线性warmup
                lr_scale = step / warmup_steps
                for param_group in opt.param_groups:
                    param_group["lr"] = 1e-3 * lr_scale
            else:
                # 余弦退火
                scheduler.step()

            step += 1
            tr_loss += loss.item() * x.size(0)
            ntr += x.size(0)

        tr_loss /= max(1, ntr)

        # 使用EMA权重进行评估
        orig_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema, strict=False)

        model.eval()
        with torch.no_grad():
            va_loss = 0.0
            nva = 0
            for x, y in va_loader:
                x = x.to(device)
                y = y.to(device)

                # 不确定性加权输出
                yhat_n = model(x)  # [B, 6P] in normalized space
                mu_n, s_n = torch.chunk(yhat_n, 2, dim=-1)  # 分离均值和方差

                # 反标准化到物理空间
                mu = mu_n * y_std + y_mean  # [B, 3P]
                y_phys = y * y_std + y_mean  # [B, 3P]

                # 1. 不确定性加权损失
                loss_hetero = weighted_finger_loss_heteroscedastic(mu_n, s_n, y)

                # 2. 骨长损失（物理空间）
                loss_bone = bone_length_loss(mu, y_phys, edges)

                # 3. 总损失组合
                loss = loss_hetero + 0.1 * loss_bone  # λ=0.1 可调

                va_loss += loss.item() * x.size(0)
                nva += x.size(0)
            va_loss /= max(1, nva)

        # 恢复原始权重
        model.load_state_dict(orig_state, strict=False)

        train_m_m, train_m_cm = compute_physical_metrics(
            model, loaders["train"], device
        )
        val_m_m, val_m_cm = compute_physical_metrics(model, loaders["val"], device)
        current_lr = opt.param_groups[0]["lr"]
        print(
            f"epoch {epoch+1:02d} | train_loss={tr_loss:.6f} val_loss={va_loss:.6f} "
            f"| train_phys={train_m_cm:.2f}cm val_phys={val_m_cm:.2f}cm | lr={current_lr:.2e}"
        )
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint")
        os.makedirs(checkpoint_path, exist_ok=True)
        if val_m_cm < best_cm - 1e-3:
            best_cm = val_m_cm
            # 保存EMA权重（更稳定的模型）
            torch.save(
                {
                    "model": ema,  # 保存EMA权重而不是当前权重
                    "dof": dof,
                    "out_dim": out_dim,
                    "x_mean": ds.x_mean,
                    "x_std": ds.x_std,
                    "y_mean": ds.y_mean,
                    "y_std": ds.y_std,
                    "hand": hand,
                    "heteroscedastic": True,  # 标记使用不确定性加权
                },
                os.path.join(checkpoint_path, "fk_mlp_best.pth"),
            )


if __name__ == "__main__":
    main()
