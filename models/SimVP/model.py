import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from .modules import ConvSC, Inception

# --- TimestepEmbedder (Copied from UNet.py) ---
class TimestepEmbedder(nn.Module):
    """
    Standard sinusoidal time embedding module
    """
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t*1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb

# --- Original SimVP Modules (Unchanged) ---

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )
    
    def forward(self,x):# B*T, C_in, H, W
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))
        
        # Output shape: (B, T*C, H, W)
        # Reshape back to (B, T, C, H, W)
        y = z.reshape(B, T, C, H, W)
        return y


# --- MODIFIED SimVP Class ---
class SimVP(nn.Module):
    def __init__(self, 
                 input_size=(6, 256, 256),  # (T, H, W)
                 in_channels_c=16,          # C_in = C_x + C_y
                 out_channels_c=8,          # C_out = C_x (noise)
                 time_emb_dim=64,           # Dimension for time embedding
                 hid_S=16, hid_T=256, N_S=4, N_T=8, 
                 incep_ker=[3,5,7,11], groups=8):
        
        super().__init__()
        
        T, H, W = input_size
        C_in = in_channels_c
        C_out = out_channels_c
        
        self.time_dim = T
        self.out_channels_c = C_out

        # 1. SimVP Backbone
        self.enc = Encoder(C_in, hid_S, N_S)
        # Mid_Xnet's input channel dim is T * hid_S
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C_out, N_S)

        # 2. Time Embeddings (from UNet.py)
        self.t_embedder = TimestepEmbedder(time_emb_dim)
        self.r_embedder = TimestepEmbedder(time_emb_dim)
        
        # 3. Projector for injecting time embedding into the bottleneck
        # We project the time_emb_dim to match the bottleneck channel dim (hid_S)
        self.time_projector = nn.Linear(time_emb_dim, hid_S)


    def forward(self, x, t, r, y=None):
        """
        Forward pass matching the UNet.py signature.
        x: (N, T, C_x, H, W) tensor of noised future video
        t: (N,) tensor of diffusion timesteps
        r: (N,) tensor of diffusion timesteps
        y: (N, T, C_y, H, W) tensor of past condition video
        """
        if y is None:
            raise ValueError("Conditional video 'y' (c_past) cannot be None for this model.")
        
        B, T, C_x, H, W = x.shape
        _, _, C_y, _, _ = y.shape
        C_in = C_x + C_y

        # 1. Calculate Time Embedding
        t_emb = self.t_embedder(t)
        r_emb = self.r_embedder(r)
        time_emb = t_emb + r_emb # (B, time_emb_dim)

        # 2. Concatenate x and y (Inputs)
        # (B, T, C_x, H, W) + (B, T, C_y, H, W) -> (B, T, C_in, H, W)
        x_inp = torch.cat([x, y], dim=2) 

        # 3. Reshape for SimVP Encoder
        # (B, T, C_in, H, W) -> (B*T, C_in, H, W)
        x_flat = x_inp.view(B*T, C_in, H, W)

        # 4. SimVP Encoder
        embed, skip = self.enc(x_flat) # embed: (B*T, C_, H_, W_)
        _, C_, H_, W_ = embed.shape

        # 5. Reshape for Mid_Xnet (Temporal)
        # (B*T, C_, H_, W_) -> (B, T, C_, H_, W_)
        z = embed.view(B, T, C_, H_, W_)
        
        # 6. --- Inject Time Embedding ---
        # Project time_emb from (B, D) to (B, C_)
        t_proj = self.time_projector(time_emb) # (B, C_)
        # Add to z: (B, T, C_, H_, W_) + (B, 1, C_, 1, 1) via broadcasting
        z = z + t_proj.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # 7. SimVP Mid_Xnet (Temporal)
        hid = self.hid(z) # (B, T, C_, H_, W_)

        # 8. Reshape for SimVP Decoder
        # (B, T, C_, H_, W_) -> (B*T, C_, H_, W_)
        hid = hid.reshape(B*T, C_, H_, W_)

        # 9. SimVP Decoder
        # Output Y shape: (B*T, C_out, H, W)
        Y = self.dec(hid, skip)
        
        # 10. --- Reshape Output (like UNet) ---
        # (B*T, C_out, H, W) -> (B, T, C_out, H, W)
        Y_3d = Y.reshape(B, self.time_dim, self.out_channels_c, H, W)
        
        # NOTE: We do NOT add the residual (Y += x_raw) because
        # 1. UNet (target) doesn't.
        # 2. Input channels (C_in) != Output channels (C_out).
        
        return Y_3d


if __name__ == "__main__":
    # --- Test harness for the new UNet-like interface ---
    B, T, H, W = 2, 6, 256, 256
    C_x, C_y = 8, 8 # C_x = future, C_y = past
    C_in = C_x + C_y # 16
    C_out = C_x      # 8

    # Create dummy inputs
    x = torch.randn([B, T, C_x, H, W])
    y_cond = torch.randn([B, T, C_y, H, W])
    t = torch.rand(B) # Timesteps (0.0 to 1.0)
    r = torch.rand(B) # Timesteps (0.0 to 1.0)
    
    print(f"--- Testing Modified SimVP ---")
    print(f"Input x shape (noised future): {x.shape}")
    print(f"Input y shape (past condition): {y_cond.shape}")
    print(f"Input t shape (timesteps): {t.shape}")

    # Initialize model with UNet-like parameters
    model = SimVP(
        input_size=(T, H, W),
        in_channels_c=C_in,
        out_channels_c=C_out
    )
    
    # Run forward pass
    output = model(x, t, r, y=y_cond)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {(B, T, C_out, H, W)}")
    
    assert output.shape == (B, T, C_out, H, W)
    print("\nTest passed! Model output shape is correct.")