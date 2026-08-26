import os
import random
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from dataset.range_transform import im_normalization
from PIL import Image 
import torch

# 设置随机种子保证可复现
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)

class CurriculumnBuilder():
    def __init__(self, im_root, video_name, model, device, total_needed, cur_epoch=-1,
                 time_lambda=0.1):
        self.vdir = os.path.join(im_root, video_name)
        self.frames = sorted(os.listdir(self.vdir))
        self.video_len = len(self.frames)
        assert self.video_len >= 3, "at least three frames"
        self.transform = transforms.Compose([transforms.ToTensor(), im_normalization])
        self.device = device
        self.model = model
        self.total_needed = total_needed
        self.id = -1
        self.course_len = self.video_len
        self.partition = total_needed
        self.cur_epoch = cur_epoch
        self.time_lambda = time_lambda
        self.reset()

    def load_frame_tensor(self, fp):
        data = np.load(fp, allow_pickle=True).item()
        img = data['image']
        pil = Image.fromarray(img)
        x = self.transform(pil).unsqueeze(0).to(self.device)
        return x
    
    def reset(self):
        interval = self.course_len - (self.cur_epoch + 1) * self.partition
        if interval <= self.partition: 
            self.id = self.course_len - self.partition - 1
        else:
            self.id = (self.cur_epoch + 1) * self.partition - 1

    def update(self, cur_epoch):
        self.cur_epoch = cur_epoch

    @torch.no_grad()
    def build_curriculum_pairs(self):
        """
        改进后的课程学习样本对构建逻辑：
        1. 先筛选所有符合跳数要求的候选对
        2. 按训练进度动态扩大帧的选取范围（从易到难）
        3. 随机采样候选对，保证样本多样性
        4. 不足时填充，满足总数要求
        """
        # 提取帧的特征并计算与帧0的难度
        zs = []
        for t in range(self.video_len):
            x = self.load_frame_tensor(os.path.join(self.vdir, self.frames[t]))
            x = x.unsqueeze(0)
            _, f16_thin, _, _, _ = self.model.encode_key(x)
            z = F.adaptive_avg_pool2d(f16_thin, 1).flatten(1)
            z = F.normalize(z, dim=1)
            zs.append(z)

        z1 = zs[0]
        diffs = []
        for t in range(1, self.video_len):
            base_diff = 1.0 - torch.sum(z1 * zs[t], dim=1).item()
            time_norm = t / (self.video_len - 1)
            d_t = base_diff + self.time_lambda * time_norm
            diffs.append((t, d_t))

        # 按难度从易到难排序
        order = [t for t, _ in sorted(diffs, key=lambda x: x[1])]

        # 阶段式调整最大跳数和帧的选取范围
        progress = (self.cur_epoch + 1) / max(1, self.partition)
        progress = min(progress, 1.0)  # 防止进度超过1
        # 1. 动态调整最大跳数（更合理的分段）
        if progress <= 0.33:
            max_jump = max(1, self.video_len // 10)
        elif progress <= 0.66:
            max_jump = max(2, self.video_len // 4)
        else:
            max_jump = min(self.video_len // 2, self.video_len - 1)  # 限制最大跳数为视频长度的一半

        # 2. 动态调整帧的选取范围（课程学习：前期只用简单帧，后期用全部）
        if progress <= 0.33:
            # 早期：只用前30%的简单帧
            selected_order = order[:int(len(order) * 0.3)]
        elif progress <= 0.66:
            # 中期：用前70%的帧
            selected_order = order[:int(len(order) * 0.7)]
        else:
            # 后期：用全部帧
            selected_order = order.copy()

        if not selected_order:
            selected_order = order  # 防止空列表

        # 3. 收集所有符合条件的候选对
        candidate_pairs = []
        # 遍历所有i<j的组合，且满足跳数要求
        for idx_i, i in enumerate(selected_order):
            # 只遍历i之后的元素，避免重复
            for j in selected_order[idx_i+1:]:
                if abs(i - j) <= max_jump:
                    candidate_pairs.append((i, j))

        # 4. 处理候选对为空的情况
        if not candidate_pairs:
            # 降级策略：生成相邻帧对
            for t in range(self.video_len - 1):
                candidate_pairs.append((t, t + 1))
            # 若仍为空，强制生成基础对
            if not candidate_pairs:
                candidate_pairs = [(0, 1)]

        # 5. 随机采样（核心改进：避免顺序选取）
        # 若候选对足够，随机选total_needed个
        if len(candidate_pairs) >= self.total_needed:
            selected_pairs = random.sample(candidate_pairs, self.total_needed)
        else:
            # 若不足，先取所有候选对，再循环填充
            selected_pairs = candidate_pairs.copy()
            while len(selected_pairs) < self.total_needed:
                # 补充时随机采样，避免重复模式
                supplement = random.sample(candidate_pairs, min(len(candidate_pairs), self.total_needed - len(selected_pairs)))
                selected_pairs.extend(supplement)

        return selected_pairs

if __name__ == "__main__":
    im_root = "/data4/jjj/video_data/ts_setv2"
    video_name = "video2"
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_needed = 1 * 10

    # 以下为原测试代码的占位实现（需根据实际环境调整）
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3, 1, 1)

        def encode_key(self, x):
            f16 = self.conv(x)
            return x, f16, None, None, None

    # 替换为实际模型（此处用dummy模型测试）
    ema_model = DummyModel().to(device).eval()
    curriculumnbuilder = CurriculumnBuilder(im_root, video_name, ema_model, device, total_needed)
    
    # 测试不同epoch的样本对生成
    for epoch in [0, 5, 10]:
        curriculumnbuilder.update(epoch)
        pairs = curriculumnbuilder.build_curriculum_pairs()
        print(f"Epoch {epoch}, Progress: {(epoch+1)/curriculumnbuilder.partition:.2f}")
        print(f"Generated pairs: {pairs}\n")