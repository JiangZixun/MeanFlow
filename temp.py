import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    # print(f"Plot saved successfully to: {save_path}")

# --- 调用示例 ---
# 假设 mock_tensor 是你的 TCHW 数据
mock_tensor = torch.randn(1, 3, 64, 64) + 2.5 # 模拟正值分布
save_distribution_plot_final(mock_tensor, "output/final_distribution.png")