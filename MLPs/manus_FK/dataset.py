# fk_data.py
import json, numpy as np, torch
from torch.utils.data import Dataset


def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵
    q: [qw, qx, qy, qz]
    """
    qw, qx, qy, qz = q

    # 归一化四元数
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # 转换为旋转矩阵
    R = np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ]
    )
    return R


def quaternion_inverse(q):
    """计算四元数的逆
    q: [qw, qx, qy, qz]
    """
    qw, qx, qy, qz = q
    norm_sq = qw * qw + qx * qx + qy * qy + qz * qz
    return np.array([qw / norm_sq, -qx / norm_sq, -qy / norm_sq, -qz / norm_sq])


def quaternion_multiply(q1, q2):
    """四元数乘法
    q1, q2: [qw, qx, qy, qz]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def transform_to_local_coordinates(poses):
    """将所有关节的坐标转换到index=0关节的局部坐标系
    poses: list of [x, y, z, qw, qx, qy, qz]
    返回: 转换后的poses
    """
    if len(poses) == 0:
        return poses

    # 获取index=0关节的位置和旋转
    origin_pos = np.array(poses[0][:3])  # [x, y, z]
    origin_quat = np.array(poses[0][3:7])  # [qw, qx, qy, qz]

    # 计算逆变换
    origin_quat_inv = quaternion_inverse(origin_quat)
    origin_rot_inv = quaternion_to_rotation_matrix(origin_quat_inv)

    transformed_poses = []

    for i, pose in enumerate(poses):
        if len(pose) < 7:
            # 如果pose数据不完整，保持原样
            transformed_poses.append(pose)
            continue

        pos = np.array(pose[:3])  # [x, y, z]
        quat = np.array(pose[3:7])  # [qw, qx, qy, qz]

        # 位置变换：先减去原点位置，再旋转
        relative_pos = pos - origin_pos
        local_pos = origin_rot_inv @ relative_pos

        # 旋转变换：q_local = q_origin_inv * q_world
        local_quat = quaternion_multiply(origin_quat_inv, quat)

        # 对于index=0关节，位置应该是[0,0,0]，旋转应该是单位四元数
        if i == 0:
            local_pos = np.array([0.0, 0.0, 0.0])
            local_quat = np.array([1.0, 0.0, 0.0, 0.0])

        transformed_poses.append(
            [
                local_pos[0],
                local_pos[1],
                local_pos[2],
                local_quat[0],
                local_quat[1],
                local_quat[2],
                local_quat[3],
            ]
        )

    return transformed_poses


class ErgoFKDataset(Dataset):
    """
    读取 jsonl，每行结构示例：
    {"ts":"...", "left": {"id":..., "angles":[...], "poses":[ [x,y,z,qw,qx,qy,qz], ... ]},
                 "right": {...}}
    我们取 hand="left"（也可以"right"），输入为 angles（float[DOF]），
    输出为所有 link 的 xyz（每个 pose 前3维），扁平化成 (num_links*3,)

    新增功能：支持将世界坐标系转换为index=0关节的局部坐标系
    """

    def __init__(
        self,
        jsonl_path,
        hand="left",
        use_links="all",
        dtype=torch.float32,
        standardize=True,
        use_local_coordinates=False,  # 新增参数：是否使用局部坐标系
    ):
        self.X, self.Y = [], []
        self.dtype = dtype
        self.standardize = standardize
        self.use_local_coordinates = use_local_coordinates

        with open(jsonl_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                h = rec.get(hand)
                if not h:
                    continue
                angles = h.get("angles")
                poses = h.get("poses")
                if angles is None or poses is None:
                    continue

                # 如果需要使用局部坐标系，先进行坐标变换
                if self.use_local_coordinates:
                    poses = transform_to_local_coordinates(poses)

                # 取 xyz（每个 pose 的前3）
                xyz = []
                for p in poses:
                    if isinstance(p, list) and len(p) >= 3:
                        xyz.extend(p[:3])
                if not angles or not xyz:
                    continue
                self.X.append(angles)
                self.Y.append(xyz)

        self.X = np.asarray(self.X, dtype=np.float32)  # [N, DOF]
        self.Y = np.asarray(self.Y, dtype=np.float32)  # [N, 3*num_links]

        # 可选：标准化，提升训练稳定性
        if self.standardize:
            self.x_mean = self.X.mean(0, keepdims=True)
            self.x_std = self.X.std(0, keepdims=True) + 1e-8
            self.y_mean = self.Y.mean(0, keepdims=True)
            self.y_std = self.Y.std(0, keepdims=True) + 1e-8
            self.Xn = (self.X - self.x_mean) / self.x_std
            self.Yn = (self.Y - self.y_mean) / self.y_std
        else:
            self.Xn, self.Yn = self.X, self.Y

    def __len__(self):
        return self.Xn.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.Xn[idx], dtype=self.dtype)  # [DOF]
        y = torch.tensor(self.Yn[idx], dtype=self.dtype)  # [3*num_links]
        return x, y
