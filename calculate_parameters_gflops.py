from models.ViT import JiT

from thop import profile
import torch
from einops import rearrange
import yaml

cfg = "configs/JiT-B_RFDPIC.yaml"

with open(cfg, 'r', encoding='utf-8') as f:
    model_config = yaml.safe_load(f)['model']


model = JiT(
    input_size=tuple(model_config['input_size']),
    in_channels_c=model_config['in_channels_c'],  # 条件输入通道数 (6)
    out_channels_c=model_config['out_channels_c'], # 预测输出通道数 (8)
    time_emb_dim=model_config['time_emb_dim'],
    patch_size=model_config['patch_size'],
    hidden_size=model_config['hidden_size'],
    depth=model_config['depth'],
    num_heads=model_config['num_heads'],
    mlp_ratio=model_config['mlp_ratio'],
    bottleneck_dim=model_config['bottleneck_dim'],
)

# 假设 model 已定义，输入尺寸为 (B, T, C, H, W)
x = torch.randn(1, 6, 8, 256, 256).cuda()
t = torch.randn(1,).cuda()
r = torch.randn(1,).cuda()
y = torch.randn(1, 6, 8, 256, 256).cuda()

model.cuda().eval()


macs, params = profile(model, inputs=(x,t,r,y), verbose=False)
flops = 2 * macs # macs 表示 Multiply-Accumulate 操作数，×2 即 FLOPs
print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
print(f"Parameters: {params/1e6:.2f} M")