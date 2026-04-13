import jax.numpy as jnp
from flax import nnx


class TimeEmbedding(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.dim = dim
        self.lin1 = nnx.Linear(dim, dim, rngs=rngs)
        self.lin2 = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, t):
        half_dim = self.dim // 2                         
        freqs = jnp.exp(jnp.arange(half_dim) * -(jnp.log(10000.0) / (half_dim - 1)))
        args = t[:, None] * freqs[None, :]                                                       
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)               
        return self.lin2(nnx.silu(self.lin1(emb)))


class UnetBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, time_dim: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features, out_features, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(out_features, out_features, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.norm = nnx.GroupNorm(num_features=out_features, num_groups=8, rngs=rngs)
        self.time_proj = nnx.Linear(time_dim, out_features, rngs=rngs)

    def __call__(self, x, t_emb):
        h = nnx.relu(self.conv1(x))                  # [B, H, W, in_features] -> [B, H, W, out_features]
        h += self.time_proj(t_emb)[:, None, None, :] # [B, out_features] -> [B, 1, 1, out_features]
        h = self.norm(h)                              # [B, H, W, out_features] -> [B, H, W, out_features]
        h = nnx.relu(self.conv2(h))                  # [B, H, W, out_features] -> [B, H, W, out_features]
        return h


class Unet(nnx.Module):
    """Instead of MaxPool, we use strides=2"""
    def __init__(self, channels: int = 3, base_dim: int = 64, time_dim: int = 256, *, rngs):

        # Encoder
        self.enc1 = UnetBlock(channels, base_dim, time_dim, rngs=rngs)
        self.down1 = nnx.Conv(base_dim, base_dim * 2, kernel_size=(3, 3), strides=2, padding="SAME", rngs=rngs)

        self.enc2 = UnetBlock(base_dim * 2, base_dim * 2, time_dim, rngs=rngs)
        self.down2 = nnx.Conv(base_dim * 2, base_dim * 4, kernel_size=(3, 3), strides=2, padding="SAME", rngs=rngs)

        # Bottleneck
        self.bottleneck = UnetBlock(base_dim * 4, base_dim * 4, time_dim, rngs=rngs)

        # Decoder
        self.up2 = nnx.ConvTranspose(base_dim * 4, base_dim * 2, kernel_size=(3, 3), strides=2, padding="SAME", rngs=rngs)
        self.dec2 = UnetBlock(base_dim * 4, base_dim * 2, time_dim, rngs=rngs)

        self.up1 = nnx.ConvTranspose(base_dim * 2, base_dim, kernel_size=(3, 3), strides=2, padding="SAME", rngs=rngs)
        self.dec1 = UnetBlock(base_dim * 2, base_dim, time_dim, rngs=rngs)

        self.final_conv = nnx.Conv(base_dim, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x, t_emb):
        # Encoder
        s1 = self.enc1(x, t_emb)                # [B, H, W, channels] -> [B, H, W, base_dim]
        h = self.down1(s1)                      # [B, H, W, base_dim] -> [B, H/2, W/2, base_dim]

        s2 = self.enc2(h, t_emb)                # [B, H/2, W/2, base_dim*2] -> [B, H/2, W/2, base_dim*2]
        h = self.down2(s2)                      # [B, H/2, W/2, base_dim*2] -> [B, H/4, W/4, base_dim*4]

        # Bottleneck
        h = self.bottleneck(h, t_emb)           # [B, H/4, W/4, base_dim*4] -> [B, H/4, W/4, base_dim*4]

        # Decoder with Skip Connections
        h = self.up2(h)                         # [B, H/4, W/4, base_dim*4] -> [B, H/2, W/2, base_dim*2]
        
        h = jnp.concatenate([h, s2], axis=-1)   # [B, H/2, W/2, base_dim*2] -> [B, H/2, W/2, base_dim*4] 
                                                # (base_dim*2 from h and base_dim*2 from s2 due to skip connection)

        h = self.dec2(h, t_emb)                 # [B, H/2, W/2, base_dim*4] -> [B, H/2, W/2, base_dim*2]

        h = self.up1(h)                         # [B, H/2, W/2, base_dim*2] -> [B, H, W, base_dim]
        h = jnp.concatenate([h, s1], axis=-1)   # [B, H, W, base_dim] -> [B, H, W, base_dim*2]
                                                # (base_dim from h and base_dim from s1 due to skip connection)        

        h = self.dec1(h, t_emb)                 # [B, H, W, base_dim*2] -> [B, H, W, base_dim]

        return self.final_conv(h)               # [B, H, W, base_dim] -> [B, H, W, channels]

class FlowMatchingModel(nnx.Module):
    def __init__(self, channels: int, base_dim: int, *, rngs: nnx.Rngs):
        self.time_dim = base_dim * 4
        self.time_map = TimeEmbedding(self.time_dim, rngs=rngs)
        self.unet = Unet(channels=channels, base_dim=base_dim, time_dim=self.time_dim, rngs=rngs)
    
    def __call__(self, x, t):
        t_emb = self.time_map(t)
        return self.unet(x, t_emb)
