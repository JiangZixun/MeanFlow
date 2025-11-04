# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage

# 全局变量
CHANNEL_NAME = [f'Ch{i+1}' for i in range(8)]
CMAP = ['gray'] * 8
label_fontsize = 12

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