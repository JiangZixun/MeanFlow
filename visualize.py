# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage
from matplotlib.colors import TABLEAU_COLORS  # 使用Tableau默认颜色集
import torch
import matplotlib
import seaborn as sns

# 全局变量
CHANNEL_NAME = ['albedo_03', 'albedo_05', 'tbb_07', 'tbb_11', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16']
CMAP = ['gray' for _ in range(len(CHANNEL_NAME))]
plot_CMAP = ['gray','inferno', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis']
# plot_CMAP = ['viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis']
label_fontsize = 16  # 放大标签字体

def vis_himawari8_seq_btchw(
        save_dir: str,
        context_seq: list,
        pred_seq: list,
        target_seq: list,
        min_max_dict=None,
    ):
    
    os.makedirs(save_dir, exist_ok=True)
    
    n_col = max(len(context_seq), len(pred_seq), len(target_seq))
    if n_col == 0:
        print("Warning: No sequences provided to visualization.")
        return
        
    row_labels = ["Context", "Pred", "Target", "Pred - Target", "Edge"]
    
    def calculate_channel_minmax(context_seq, target_seq, channel_idx):
        all_values = []
        for seq in [context_seq, target_seq]:
            if seq:
                for frame in seq:
                    if frame is not None:
                        if hasattr(frame, 'detach'):
                            frame_np = frame.detach().cpu().numpy()
                        else:
                            frame_np = frame
                        
                        if channel_idx < frame_np.shape[0]:
                             all_values.append(frame_np[channel_idx, ...].flatten())
        
        if all_values:
            combined_values = np.concatenate(all_values)
            return combined_values.min(), combined_values.max()
        else:
            return 0, 1
    
    for channel_idx in range(8):
        plt.figure(figsize=(n_col*2, 5 * 1.8))
        
        if min_max_dict is None:
            vmin, vmax = calculate_channel_minmax(context_seq, target_seq, channel_idx)
        else:
            vmin = min_max_dict['min'][channel_idx]
            vmax = min_max_dict['max'][channel_idx]
        
        gs = GridSpec(5, n_col+1, figure=plt.gcf(),
                    width_ratios=[0.5]+[1]*n_col,
                    wspace=0.05, hspace=0.05,
                    left=0, right=1, bottom=0, top=1)
        
        for row in range(5):
            ax_label = plt.subplot(gs[row, 0])
            ax_label.text(1, 0.5, row_labels[row], 
                        ha='right', va='center',
                        fontsize=label_fontsize, rotation=90)
            ax_label.axis('off')
        
        def get_img_np(frame, ch_idx):
            if hasattr(frame, 'detach'):
                frame = frame.detach().cpu().numpy()
            if ch_idx < frame.shape[0]:
                return frame[ch_idx, ...]
            else:
                return np.zeros((256, 256))

        for t in range(len(context_seq)):
            ax = plt.subplot(gs[0, t+1])
            img = get_img_np(context_seq[t], channel_idx)
            ax.imshow(img, cmap=CMAP[channel_idx], vmin=vmin, vmax=vmax)
            ax.axis('off')
        
        for t in range(len(pred_seq)):
            ax = plt.subplot(gs[1, t+1])
            img = get_img_np(pred_seq[t], channel_idx)
            ax.imshow(img, cmap=CMAP[channel_idx], vmin=vmin, vmax=vmax)
            ax.axis('off')
        
        for t in range(len(target_seq)):
            ax = plt.subplot(gs[2, t+1])
            img = get_img_np(target_seq[t], channel_idx)
            ax.imshow(img, cmap=CMAP[channel_idx], vmin=vmin, vmax=vmax)
            ax.axis('off')
        
        diff_values = []
        valid_diff_frames = 0
        if len(pred_seq) > 0 and len(target_seq) > 0:
            for t in range(min(len(pred_seq), len(target_seq))):
                pred_img = get_img_np(pred_seq[t], channel_idx)
                target_img = get_img_np(target_seq[t], channel_idx)
                diff = pred_img - target_img
                diff_values.append(diff.flatten())
                valid_diff_frames += 1

            if diff_values:
                combined_diff = np.concatenate(diff_values)
                diff_vmin, diff_vmax = combined_diff.min(), combined_diff.max()
                diff_abs_max = max(abs(diff_vmin), abs(diff_vmax))
                diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max
            else:
                diff_vmin, diff_vmax = -1, 1
        else:
            diff_vmin, diff_vmax = -1, 1
        
        for t in range(valid_diff_frames):
            ax = plt.subplot(gs[3, t+1])
            pred_img = get_img_np(pred_seq[t], channel_idx)
            target_img = get_img_np(target_seq[t], channel_idx)
            diff = pred_img - target_img
            ax.imshow(diff, cmap='coolwarm', vmin=diff_vmin, vmax=diff_vmax)
            ax.axis('off')
        
        def apply_edge_detection(img):
            if hasattr(img, 'detach'):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img
            
            img_smooth = ndimage.gaussian_filter(img_np, sigma=0.1)
            grad_x = ndimage.sobel(img_smooth, axis=1)
            grad_y = ndimage.sobel(img_smooth, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            threshold = np.percentile(gradient_magnitude, 80)
            edge_mask = gradient_magnitude > threshold
            binary_edge = edge_mask.astype(np.float32)
            return binary_edge
        
        edge_vmin, edge_vmax = 0, 1
        for t in range(len(pred_seq)):
            ax = plt.subplot(gs[4, t+1])
            edge_img = apply_edge_detection(get_img_np(pred_seq[t], channel_idx))
            ax.imshow(edge_img, cmap='gray', vmin=edge_vmin, vmax=edge_vmax)
            ax.axis('off')
        
        save_path = os.path.join(save_dir, f"{CHANNEL_NAME[channel_idx]}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close('all') # 关闭所有图像，防止内存泄漏


def plot_metrics_curve(save_dir:str,
                       name:str,
                       data: np.ndarray):
    """
    绘制指标随时间变化的曲线图，分为两组：
    1. albedo_03和albedo_05
    2. 其他6个tbb通道
    参数:
        save_dir: 图片保存目录
        name: 指标名称 (如'MSE', 'MAE')
        data: 8x6的numpy数组，每行代表一个通道，每列代表一个时间步
    """
    os.makedirs(save_dir, exist_ok=True)
    # 时间设置（30分钟间隔）
    time_labels = [f'{(i+1)*30}min' for i in range(6)]  # 00:00, 00:30, ..., 02:30
    x_ticks = np.arange(6)
    
    # ===================== 第一张图：两个albedo通道 =====================
    plt.figure(figsize=(10, 5))
    albedo_indices = [0, 1]  # albedo_03和albedo_05的索引
    for i in albedo_indices:
        plt.plot(x_ticks, data[i], 
                color=list(TABLEAU_COLORS.values())[i],
                marker='o',
                linewidth=2,
                label=CHANNEL_NAME[i])
    plt.title(f'{name} - Albedo Channels')
    plt.ylabel(name)
    plt.xticks(x_ticks, time_labels)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    # 保存albedo专用图
    albedo_path = os.path.join(save_dir, f'{name.lower()}_albedo.png')
    plt.savefig(albedo_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===================== 第二张图：6个tbb通道 =====================
    plt.figure(figsize=(12, 6))
    tbb_indices = [2, 3, 4, 5, 6, 7]  # 其他6个tbb通道的索引
    for i in tbb_indices:
        plt.plot(x_ticks, data[i], 
                color=list(TABLEAU_COLORS.values())[i],
                marker='o',
                linewidth=2,
                label=CHANNEL_NAME[i])
    plt.title(f'{name} - TBB Channels')
    plt.ylabel(name)
    plt.xticks(x_ticks, time_labels)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    # 调整布局防止标签重叠
    plt.tight_layout()
    # 保存tbb专用图
    tbb_path = os.path.join(save_dir, f'{name.lower()}_tbb.png')
    plt.savefig(tbb_path, dpi=300, bbox_inches='tight')
    plt.close()


def vis_dynamics_feats(save_dir, dyn_feats, batch_idx=0):
    """
    可视化动力学特征并按阶段分类保存
    :param save_dir: 根保存路径
    :param dyn_feats: 包含 div, vort, speed 等特征的字典, 形状为 (B, H, W) 或 (B, 1, H, W)
    :param batch_idx: 指定可视化的 Batch 索引
    """
    # 1. 定义阶段映射 (对应你代码中的 proj 分组逻辑)
    stages = {
        "early": ["speed", "e1", "e2", "shear"],
        "mid":   ["vort"],
        "late":  ["div", "strain"]
    }
    
    # 2. 设置颜色映射 (根据物理意义选择)
    # 散度、涡度、应变分量通常有正负，使用离散色彩蓝-红 (seismic/coolwarm)
    # 速率、剪切力、总应变只有正值，使用感知均匀的色彩 (viridis/magma)
    cmap_dict = {
        "div": "RdBu_r", "vort": "RdBu_r", "e1": "RdBu_r", "e2": "RdBu_r",
        "speed": "magma", "shear": "viridis", "strain": "viridis"
    }

    # 3. 遍历阶段进行绘图
    for stage_name, feat_list in stages.items():
        # 创建第二层文件夹: save_dir/early, save_dir/mid, etc.
        stage_dir = os.path.join(save_dir, stage_name)
        os.makedirs(stage_dir, exist_ok=True)

        for feat_name in feat_list:
            if feat_name not in dyn_feats:
                continue
            
            # 提取张量并转为 Numpy
            feat_tensor = dyn_feats[feat_name]
            
            # 处理维度 (B, H, W) 或 (B, 1, H, W) -> (H, W)
            if feat_tensor.ndim == 4:
                img = feat_tensor[batch_idx, 0].detach().cpu().numpy()
            else:
                img = feat_tensor[batch_idx].detach().cpu().numpy()

            # 绘图
            plt.figure(figsize=(6, 6))
            
            # 对对称物理量(div, vort, e1, e2)设置对称的颜色刻度中心点为0
            if cmap_dict[feat_name] == "RdBu_r":
                vabs = np.max(np.abs(img))
                vmin, vmax = -vabs, vabs
            else:
                vmin, vmax = np.min(img), np.max(img)

            im = plt.imshow(img, cmap=cmap_dict[feat_name], vmin=vmin, vmax=vmax)
            
            # 仿照你的代码风格：去掉坐标轴，增加颜色条和标题
            plt.axis('off')
            # plt.title(f"{feat_name.upper()} ({stage_name})", fontsize=12)
            # plt.colorbar(im, fraction=0.046, pad=0.04)
            
            # 保存路径: save_dir/{stage}/{feat_name}.png
            save_path = os.path.join(stage_dir, f"{feat_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()


# def visualize_channel_fields(
#         displacement_fields,                 # [T, C, H, W, 2]
#         save_dir,
#         field_type="displacement",           # or "velocity" / "masked_displacement"
#         size=4,                              # 采样步长
#         show=False
#     ):
#     assert field_type in {"displacement", "velocity", "masked_displacement"}, \
#         f"field_type must be displacement / velocity / masked_displacement, got {field_type}"
#     os.makedirs(save_dir, exist_ok=True)

#     T, C, H, W, _ = displacement_fields.shape

#     for c in range(C):
#         fig, axes = plt.subplots(1, T, figsize=(T * 3, 3))
#         axes = np.atleast_1d(axes)           # 兼容 T==1 情况

#         # 每个时间步画一格
#         for t in range(T):
#             vel = displacement_fields[t, c].cpu().numpy()   # [H,W,2]

#             y, x = np.mgrid[0:H:size, 0:W:size]
#             u = -vel[::size, ::size, 0]
#             v = -vel[::size, ::size, 1]
#             magnitude = np.sqrt(u**2 + v**2)

#             ax = axes[t]
#             q = ax.quiver(
#                 x, y, u, v, magnitude,
#                 cmap="viridis", alpha=0.9, width=0.003
#             )
#             ax.set_title(f"t={t}")
#             ax.axis("off")
#             ax.set_xlim(0, W)
#             ax.set_ylim(H, 0)

#         plt.tight_layout()
#         fname = f"{field_type}_{CHANNEL_NAME[c]}"
#         plt.savefig(os.path.join(save_dir, f"{fname}.png"),
#                     dpi=300, bbox_inches="tight")
#         if show:
#             plt.show()
#         plt.close()

def visualize_channel_fields(
        displacement_fields,                 # [T, C, H, W, 2]
        save_dir,
        field_type="displacement",           # or "velocity" / "masked_displacement"
        size=4,                              # 采样步长
        show=False
    ):
    assert field_type in {"displacement", "velocity", "masked_displacement"}, \
        f"field_type must be displacement / velocity / masked_displacement, got {field_type}"
    
    T, C, H, W, _ = displacement_fields.shape

    # 遍历每个通道
    for c in range(C):
        # 为当前通道创建子文件夹
        # 假设 CHANNEL_NAME 是全局定义的列表或字典
        channel_name = CHANNEL_NAME[c]
        channel_save_path = os.path.join(save_dir, channel_name)
        os.makedirs(channel_save_path, exist_ok=True)

        # 遍历每个时间步
        for t in range(T):
            # 创建单张画布
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # 提取数据并转为 numpy
            vel = displacement_fields[t, c].cpu().numpy()   # [H, W, 2]

            # 生成网格
            y, x = np.mgrid[0:H:size, 0:W:size]
            u = -vel[::size, ::size, 0]
            v = -vel[::size, ::size, 1]
            magnitude = np.sqrt(u**2 + v**2)

            # 绘制矢量场
            q = ax.quiver(
                x, y, u, v, magnitude,
                cmap="viridis", alpha=0.9, width=0.003
            )
            
            # 美化设置
            # ax.set_title(f"{field_type} | {channel_name} | t={t}")
            ax.axis("off")
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0) # 保持坐标轴方向与图像一致

            # 保存当前时间步的文件
            file_name = f"{field_type}_t{t:03d}.png"
            plt.savefig(
                os.path.join(channel_save_path, file_name),
                dpi=300, 
                bbox_inches="tight"
            )
            
            if show:
                plt.show()
            
            plt.close(fig) # 及时关闭释放内存

    # print(f"可视化完成，结果保存在: {save_dir}")


# def visualize_channel_residuals(
#     residual_fields,              # [T,C,H,W] 或 [B,T,C,H,W]
#     save_dir,
#     cmap="RdBu_r",
#     vmin=None, vmax=None,         # 若为 None，按分位数自动估计并做对称色标
#     qmin=1.0, qmax=99.0,
#     per_channel_scale=True,       # True: 每通道独立色标；False: 全通道共享
#     show=False
# ):
#     """
#     把标量残差场按通道绘图，每张图横向排 T 个时刻；颜色条独立 cax 放在最右侧。
#     文件名: residual_<channel_name>.png
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # ---- 统一为 numpy，形状 [T,C,H,W] ----
#     if isinstance(residual_fields, torch.Tensor):
#         residual_fields = residual_fields.detach().cpu().numpy()
#     if residual_fields.ndim == 5:
#         residual_fields = residual_fields[0]             # 取第 0 个 batch
#     assert residual_fields.ndim == 4, f"expect [T,C,H,W], got {residual_fields.shape}"

#     T, C, H, W = residual_fields.shape

#     # 通道名
#     try:
#         names = CHANNEL_NAME
#         assert len(names) >= C
#     except Exception:
#         names = [f"ch{c}" for c in range(C)]

#     # 若使用全局共享色标，预先计算范围
#     if not per_channel_scale:
#         if vmin is None or vmax is None:
#             lo = np.percentile(residual_fields, qmin)
#             hi = np.percentile(residual_fields, qmax)
#             M = max(abs(lo), abs(hi))
#             gvmin, gvmax = -M, M
#         else:
#             gvmin, gvmax = vmin, vmax

#     for c in range(C):
#         fig, axes = plt.subplots(1, T, figsize=(T * 3, 3))
#         axes = np.atleast_1d(axes)

#         # 该通道的色标范围
#         if per_channel_scale:
#             data_c = residual_fields[:, c]               # [T,H,W]
#             if vmin is None or vmax is None:
#                 lo = np.percentile(data_c, qmin)
#                 hi = np.percentile(data_c, qmax)
#                 M = max(abs(lo), abs(hi))
#                 cvmin, cvmax = -M, M
#             else:
#                 cvmin, cvmax = vmin, vmax
#         else:
#             cvmin, cvmax = gvmin, gvmax

#         # 逐时刻绘图
#         for t in range(T):
#             img = residual_fields[t, c]
#             ax = axes[t]
#             im = ax.imshow(img, cmap=cmap, vmin=cvmin, vmax=cvmax)
#             ax.set_title(f"t={t}")
#             ax.axis("off")

#         # —— 独立 cax：自动贴到最右一列子图之外 —— #
#         # 用 ScalarMappable 构建色条，不依赖最后一幅图
#         sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=cvmin, vmax=cvmax))
#         sm.set_array([])

#         # 计算所有子图的边界，自动确定 cax
#         fig.canvas.draw()  # 触发布局，确保位置可用
#         pos = [ax.get_position() for ax in axes.ravel()]
#         right  = max(p.x1 for p in pos)
#         left   = right + 0.01         # 与子图间距
#         bottom = min(p.y0 for p in pos)
#         top    = max(p.y1 for p in pos)
#         height = top - bottom
#         width  = 0.015                # 色条宽度

#         cax = fig.add_axes([left, bottom, width, height])
#         cbar = fig.colorbar(sm, cax=cax)
#         cbar.set_label("residual value")

#         plt.tight_layout(rect=[0, 0, right, 1])  # 避免挤压到 cax
#         fname = f"residual_{names[c]}"
#         fig.savefig(os.path.join(save_dir, f"{fname}.png"), dpi=300, bbox_inches="tight")
#         if show:
#             plt.show()
#         plt.close(fig)

def visualize_channel_residuals(
    residual_fields,              # [T,C,H,W] 或 [B,T,C,H,W]
    save_dir,
    cmap="RdBu_r",
    vmin=None, vmax=None,         # 若为 None，按分位数自动估计并做对称色标
    qmin=1.0, qmax=99.0,
    per_channel_scale=True,       # True: 每通道独立色标；False: 全通道共享
    show=False
):
    """
    将标量残差场按通道和时间步保存为独立文件。
    目录结构: save_dir/channel_name/residual_t000.png
    """
    # ---- 统一为 numpy，形状 [T,C,H,W] ----
    if isinstance(residual_fields, torch.Tensor):
        residual_fields = residual_fields.detach().cpu().numpy()
    if residual_fields.ndim == 5:
        residual_fields = residual_fields[0]             
    assert residual_fields.ndim == 4, f"expect [T,C,H,W], got {residual_fields.shape}"

    T, C, H, W = residual_fields.shape

    # 通道名处理
    try:
        names = CHANNEL_NAME
    except NameError:
        names = [f"ch{c}" for c in range(C)]

    # 全局色标计算
    if not per_channel_scale:
        if vmin is None or vmax is None:
            lo = np.percentile(residual_fields, qmin)
            hi = np.percentile(residual_fields, qmax)
            M = max(abs(lo), abs(hi))
            gvmin, gvmax = -M, M
        else:
            gvmin, gvmax = vmin, vmax

    for c in range(C):
        # 创建通道子文件夹
        channel_name = names[c]
        channel_save_path = os.path.join(save_dir, channel_name)
        os.makedirs(channel_save_path, exist_ok=True)

        # 确定该通道的色标范围（确保该通道内所有时间步使用同一尺度）
        if per_channel_scale:
            data_c = residual_fields[:, c]
            if vmin is None or vmax is None:
                lo = np.percentile(data_c, qmin)
                hi = np.percentile(data_c, qmax)
                M = max(abs(lo), abs(hi))
                cvmin, cvmax = -M, M
            else:
                cvmin, cvmax = vmin, vmax
        else:
            cvmin, cvmax = gvmin, gvmax

        # 逐时刻保存为独立文件
        for t in range(T):
            fig, ax = plt.subplots(figsize=(5, 5))
            
            img_data = residual_fields[t, c]
            im = ax.imshow(img_data, cmap=cmap, vmin=cvmin, vmax=cvmax)
            
            # ax.set_title(f"{channel_name} | t={t}")
            ax.axis("off")

            # # 在单张图侧边添加 colorbar
            # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # cbar.set_label("residual value")

            # 保存
            fname = f"residual_t{t:03d}.png"
            plt.savefig(
                os.path.join(channel_save_path, fname), 
                dpi=300, 
                bbox_inches="tight"
            )
            
            if show:
                plt.show()
            plt.close(fig)

    # print(f"残差场可视化完成，已保存至: {save_dir}")


import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# def plot_himawari8_seq_btchw(
#         save_dir: str,
#         context_seq: list,
#         pred_dict: dict,  # 修改为字典: {'deterministic': pred_seq1, 'flowmatching': pred_seq2}
#         target_seq: list,
#     ):
#     """
#     可视化Himawari8序列，包含确定性模型和Flow Matching模型的预测结果。
#     """

#     # 动态计算每个通道的min/max值（只基于context和target序列，确保对比公平）
#     def calculate_channel_minmax(context_seq, target_seq, channel_idx):
#         all_values = []
#         for seq in [context_seq, target_seq]:
#             if seq:
#                 for frame in seq:
#                     if frame is not None:
#                         # 处理 PyTorch Tensor 或 Numpy
#                         frame_np = frame.detach().cpu().numpy() if hasattr(frame, 'detach') else frame
#                         all_values.append(frame_np[channel_idx, ...].flatten())
        
#         if all_values:
#             combined_values = np.concatenate(all_values)
#             return combined_values.min(), combined_values.max()
#         return 0, 1
    
#     # 预计算归一化范围
#     min_max_dict = {'min': [], 'max': []}
#     for channel_idx in range(8):
#         min_val, max_val = calculate_channel_minmax(context_seq, target_seq, channel_idx)
#         min_max_dict['min'].append(min_val)
#         min_max_dict['max'].append(max_val)
    
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 定义需要循环处理的序列任务
#     # context 和 target 是单序列，pred_dict 包含两个预测序列
#     for channel_idx in range(8):
#         channel_name = CHANNEL_NAME[channel_idx]
#         channel_dir = os.path.join(save_dir, channel_name)
#         vmin = min_max_dict['min'][channel_idx]
#         vmax = min_max_dict['max'][channel_idx]
#         cmap = plot_CMAP[channel_idx]

#         # 整理所有待绘图的任务：(子目录名, 对应的序列数据)
#         tasks = [
#             ('context', context_seq),
#             ('target', target_seq)
#         ]
#         # 将两种预测模型加入任务列表
#         for model_name, seq in pred_dict.items():
#             tasks.append((f'pred_{model_name}', seq))

#         for folder_name, sequence in tasks:
#             if sequence is None: continue
            
#             # 创建子目录: 例如 channel_name/pred_deterministic/
#             type_dir = os.path.join(channel_dir, folder_name)
#             os.makedirs(type_dir, exist_ok=True)
            
#             for t in range(len(sequence)):
#                 fig, ax = plt.subplots(figsize=(8, 8))
                
#                 img = sequence[t][channel_idx, ...]
#                 # 如果是Tensor，转为numpy
#                 if hasattr(img, 'detach'):
#                     img = img.detach().cpu().numpy()
                
#                 ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
#                 ax.axis('off')
                
#                 # 紧凑布局保存
#                 plt.tight_layout()
#                 plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
#                 save_path = os.path.join(type_dir, f'time_step_{t:01d}.png')
#                 plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
#                 plt.close(fig)

#     # print(f"可视化完成，结果保存至: {save_dir}")

class SobelGrad(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        gy = gx.t()
        self.register_buffer('kx', gx.view(1,1,3,3))
        self.register_buffer('ky', gy.view(1,1,3,3))
        self.channels = channels

    def forward(self, x):
        # x shape: (C, H, W) -> 需要增加 Batch 维度 (1, C, H, W)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        kx = self.kx.repeat(self.channels, 1, 1, 1)
        ky = self.ky.repeat(self.channels, 1, 1, 1)
        grad_x = F.conv2d(x, kx, padding=1, groups=self.channels)
        grad_y = F.conv2d(x, ky, padding=1, groups=self.channels)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        return grad_mag.squeeze(0) # 返回 (C, H, W)

def calculate_robust_minmax(seq_list, channel_idx, p_min=2, p_max=98):
    all_values = []
    for seq in seq_list:
        if seq is not None:
            for frame in seq:
                frame_np = frame.detach().cpu().numpy() if hasattr(frame, 'detach') else frame
                all_values.append(frame_np[channel_idx, ...].flatten())
    
    if all_values:
        combined_values = np.concatenate(all_values)
        # 使用分位数代替绝对值，显著提升对比度
        vmin = np.percentile(combined_values, p_min)
        vmax = np.percentile(combined_values, p_max)
        return vmin, vmax
    return 0, 1

def plot_himawari8_seq_btchw(
        save_dir: str,
        context_seq: list,
        pred_dict: dict,
        target_seq: list,
    ):
    """
    可视化Himawari8序列，包含原始图像和Sobel梯度图像。
    """
    # 实例化梯度提取器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sobel_node = SobelGrad(channels=8).to(device)
    
    def to_tensor(frame):
        if not isinstance(frame, torch.Tensor):
            return torch.from_numpy(frame).to(device)
        return frame.to(device)

    # --- 新增：计算所有序列的梯度序列 ---
    def get_grad_seq(seq):
        if seq is None: return None
        grad_seq = []
        for frame in seq:
            with torch.no_grad():
                # 确保输入是 Tensor 并计算梯度
                frame_t = to_tensor(frame)
                grad_mag = sobel_node(frame_t)
                grad_seq.append(grad_mag.cpu().numpy())
        return grad_seq

    context_grad = get_grad_seq(context_seq)
    target_grad = get_grad_seq(target_seq)
    pred_grad_dict = {f"{k}_grad": get_grad_seq(v) for k, v in pred_dict.items()}

    # 修改 min/max 计算逻辑，支持梯度序列的归一化
    def calculate_channel_minmax(seq_list, channel_idx):
        all_values = []
        for seq in seq_list:
            if seq is not None:
                for frame in seq:
                    # 此时 frame 已经是 numpy (来自原始逻辑或上面的 grad 计算)
                    frame_np = frame.detach().cpu().numpy() if hasattr(frame, 'detach') else frame
                    all_values.append(frame_np[channel_idx, ...].flatten())
        
        if all_values:
            combined_values = np.concatenate(all_values)
            return combined_values.min(), combined_values.max()
        return 0, 1

    # 预计算：原始图像范围
    min_max_raw = {'min': [], 'max': []}
    # 预计算：梯度图像范围
    min_max_grad = {'min': [], 'max': []}

    for channel_idx in range(8):
        # 原始图像范围计算
        mi, ma = calculate_channel_minmax([context_seq, target_seq], channel_idx)
        min_max_raw['min'].append(mi)
        min_max_raw['max'].append(ma)
        
        # 梯度图改用鲁棒计算 (针对边缘增强)
        mi_g, ma_g = calculate_robust_minmax([context_grad, target_grad], channel_idx, p_min=1, p_max=99)
        min_max_grad['min'].append(mi_g)
        min_max_grad['max'].append(ma_g)

    os.makedirs(save_dir, exist_ok=True)
    
    for channel_idx in range(8):
        channel_name = CHANNEL_NAME[channel_idx]
        channel_dir = os.path.join(save_dir, channel_name)
        cmap = plot_CMAP[channel_idx]

        # 整理所有待绘图的任务：(子目录名, 序列数据, 是否是梯度)
        tasks = [
            ('context', context_seq, False),
            ('target', target_seq, False),
            ('context_grad', context_grad, True),
            ('target_grad', target_grad, True)
        ]
        
        for model_name, seq in pred_dict.items():
            tasks.append((f'pred_{model_name}', seq, False))
        for model_name, grad_seq in pred_grad_dict.items():
            tasks.append((f'{model_name}', grad_seq, True))

        for folder_name, sequence, is_grad in tasks:
            if sequence is None: continue
            
            type_dir = os.path.join(channel_dir, folder_name)
            os.makedirs(type_dir, exist_ok=True)
            
            # 根据是否是梯度选择对应的归一化范围
            if is_grad:
                vmin, vmax = min_max_grad['min'][channel_idx], min_max_grad['max'][channel_idx]
                # 梯度通常使用灰度或特定的感知色彩，如果想用原始色图也可以
                current_cmap = 'magma' # 或者保留 cmap = cmap
            else:
                vmin, vmax = min_max_raw['min'][channel_idx], min_max_raw['max'][channel_idx]
            
            for t in range(len(sequence)):
                fig, ax = plt.subplots(figsize=(8, 8))
                
                img = sequence[t][channel_idx, ...]
                if hasattr(img, 'detach'):
                    img = img.detach().cpu().numpy()
                
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis('off')
                
                plt.tight_layout()
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
                save_path = os.path.join(type_dir, f'time_step_{t:01d}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close(fig)


def save_distribution_plot_final(tensor, save_path):
    """
    绘制最终版的概率分布图：蓝色曲线，轴线曲线等粗，
    L型坐标轴（无负轴），无刻度，带精准箭头。
    """
    # 1. 数据处理：展平 Tensor
    data = tensor.detach().cpu().numpy().flatten()
    
    # 定义全局视觉参数
    lw = 2.0             # 统一粗细
    blue_color = '#1f77b4' # 蓝色曲线
    
    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 3. 绘制平滑曲线 (KDE)
    sns.kdeplot(data, ax=ax, color=blue_color, linewidth=lw, fill=True, alpha=0.15)
    
    # --- 4. 坐标轴美化 (L型，移除负轴感) ---
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # 设置坐标轴起始位置 (x轴起始于数据最小值，y轴起始于0)
    x_start = data.min()
    ax.spines['left'].set_position(('data', x_start))
    ax.spines['bottom'].set_position(('data', 0))
    
    # 设置轴线粗细
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    
    # 彻底移除刻度和刻度线
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 5. 动态设置范围并为箭头预留空间
    ax.relim()
    ax.autoscale_view()
    x_min, x_max = ax.get_xlim()
    _, y_max = ax.get_ylim()
    
    # 扩展范围以确保箭头在曲线外侧
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.1)
    ax.set_ylim(0, y_max * 1.2)
    
    # --- 6. 绘制精准的坐标轴箭头 ---
    # 使用标注功能在轴的最末端添加箭头，确保位置和方向准确
    # X轴箭头
    ax.annotate('', xy=(1.03, 0), xycoords=('axes fraction', 'data'), 
                xytext=(1.0, 0), textcoords=('axes fraction', 'data'),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=lw, mutation_scale=20))
    
    # Y轴箭头
    ax.annotate('', xy=(x_start, 1.05), xycoords=('data', 'axes fraction'), 
                xytext=(x_start, 1.0), textcoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=lw, mutation_scale=20))
    
    # --- 7. 设置数学标签 ---
    # 显式清除 Seaborn 自动生成的 "Density" 标签
    ax.set_ylabel('')
    ax.set_xlabel('') # 如果 X 轴也有默认标签也可一并清除

    # 使用 annotate 精准放置自定义标签
    # Y轴标签 $p(x)$：放在箭头左上方
    ax.annotate('$p(x)$', xy=(x_start, 1.05), xycoords=('data', 'axes fraction'),
                xytext=(-10, 5), textcoords='offset points', 
                ha='right', va='center', fontsize=14)
    
    # X轴标签 $x$：放在箭头右下方
    ax.annotate('$x$', xy=(1.03, 0), xycoords=('axes fraction', 'data'),
                xytext=(5, -15), textcoords='offset points', 
                ha='center', va='top', fontsize=14)
    
    # 8. 保存图片
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    plt.tight_layout()
    # 保存为透明背景，方便放入论文或 PPT
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()