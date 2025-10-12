import jax.numpy as jnp
from flax import nnx


def depth_to_space(x, block_size=2):
    """Converts depth to spatial dimensions (pixel shuffle) for NHWC format."""
    B, H, W, C = x.shape
    r = block_size
    out_c = C // (r * r)

    # Reshape: [B, H, W, C] -> [B, H, W, r, r, out_c]
    x = x.reshape(B, H, W, r, r, out_c)
    # Transpose: [B, H, W, r, r, out_c] -> [B, H, r, W, r, out_c]
    x = x.transpose(0, 1, 3, 2, 4, 5)
    # Reshape: [B, H, r, W, r, out_c] -> [B, H*r, W*r, out_c]
    x = x.reshape(B, H * r, W * r, out_c)
    return x


def space_to_depth(x, block_size=2):
    """Converts spatial dimensions to depth for NHWC format."""
    B, H, W, C = x.shape
    r = block_size
    out_c = C * (r * r)

    # Reshape: [B, H, W, C] -> [B, H//r, r, W//r, r, C]
    x = x.reshape(B, H // r, r, W // r, r, C)
    # Transpose: [B, H//r, r, W//r, r, C] -> [B, H//r, W//r, r, r, C]
    x = x.transpose(0, 1, 3, 2, 4, 5)
    # Reshape: [B, H//r, W//r, r, r, C] -> [B, H//r, W//r, out_c]
    x = x.reshape(B, H // r, W // r, out_c)
    return x


class ResBlockUp(nnx.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        rngs: nnx.Rngs,
        use_bn: bool = True,
        use_sn: bool = False,
    ):
        self.use_bn = use_bn
        self.out_ch = out_ch

        # After upsampling via depth_to_space, channels are divided by 4
        upsampled_ch = in_ch // 4

        if use_sn:
            from utils.spectral_norm import SpectralNormConv

            self.conv1 = SpectralNormConv(
                out_ch, kernel_size=(3, 3), padding="SAME", rngs=rngs
            )
            self.conv2 = SpectralNormConv(
                out_ch, kernel_size=(3, 3), padding="SAME", rngs=rngs
            )
            self.conv_shortcut = SpectralNormConv(
                out_ch, kernel_size=(1, 1), padding="SAME", rngs=rngs
            )
        else:
            self.conv1 = nnx.Conv(
                in_features=upsampled_ch,
                out_features=out_ch,
                kernel_size=(3, 3),
                padding="SAME",
                rngs=rngs,
            )
            self.conv2 = nnx.Conv(
                in_features=out_ch,
                out_features=out_ch,
                kernel_size=(3, 3),
                padding="SAME",
                rngs=rngs,
            )
            self.conv_shortcut = nnx.Conv(
                in_features=upsampled_ch,
                out_features=out_ch,
                kernel_size=(1, 1),
                padding="SAME",
                rngs=rngs,
            )

        if use_bn:
            self.bn1 = nnx.BatchNorm(in_ch, rngs=rngs)
            self.bn2 = nnx.BatchNorm(out_ch, rngs=rngs)

    def __call__(self, x):
        residual = x
        # Main path
        if self.use_bn:
            x = self.bn1(x)
        x = nnx.relu(x)
        x = depth_to_space(x)  # Upsample: [B, H, W, C] -> [B, H*2, W*2, C//4]
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn2(x)
        x = nnx.relu(x)
        x = self.conv2(x)

        shortcut = depth_to_space(residual)
        shortcut = self.conv_shortcut(shortcut)

        return x + shortcut


class ResBlockDown(nnx.Module):
    def __init__(
        self, in_ch: int, out_ch: int, *, rngs: nnx.Rngs, use_sn: bool = False
    ):
        self.out_ch = out_ch

        # After space_to_depth on input, channels are in_ch * 4
        # After conv1 then space_to_depth, channels are out_ch * 4
        in_ch_after_downsample = in_ch * 4
        out_ch_after_downsample = out_ch * 4

        if use_sn:
            from utils.spectral_norm import SpectralNormConv

            self.conv1 = SpectralNormConv(
                out_ch, kernel_size=(3, 3), padding="SAME", rngs=rngs
            )
            self.conv2 = SpectralNormConv(
                out_ch, kernel_size=(3, 3), padding="SAME", rngs=rngs
            )
            self.conv_shortcut = SpectralNormConv(
                out_ch, kernel_size=(1, 1), padding="SAME", rngs=rngs
            )
        else:
            self.conv1 = nnx.Conv(
                in_features=in_ch,
                out_features=out_ch,
                kernel_size=(3, 3),
                padding="SAME",
                rngs=rngs,
            )
            self.conv2 = nnx.Conv(
                in_features=out_ch_after_downsample,
                out_features=out_ch,
                kernel_size=(3, 3),
                padding="SAME",
                rngs=rngs,
            )
            self.conv_shortcut = nnx.Conv(
                in_features=in_ch_after_downsample,
                out_features=out_ch,
                kernel_size=(1, 1),
                padding="SAME",
                rngs=rngs,
            )

    def __call__(self, x):
        residual = x
        x = nnx.relu(x)
        x = self.conv1(x)
        x = space_to_depth(x)  # Downsample: [B, H, W, C] -> [B, H//2, W//2, C*4]
        x = self.conv2(x)

        # Shortcut
        shortcut = space_to_depth(residual)
        shortcut = self.conv_shortcut(shortcut)

        return x + shortcut


class Generator(nnx.Module):
    def __init__(self, latent_dim: int = 128, n_filters: int = 128, *, rngs: nnx.Rngs):
        self.latent_dim = latent_dim
        self.n_filters = n_filters

        self.dense = nnx.Linear(
            in_features=latent_dim, out_features=4 * 4 * 256, rngs=rngs
        )
        self.resblock1 = ResBlockUp(256, n_filters, rngs=rngs, use_bn=True)
        self.resblock2 = ResBlockUp(n_filters, n_filters, rngs=rngs, use_bn=True)
        self.resblock3 = ResBlockUp(n_filters, n_filters, rngs=rngs, use_bn=True)
        self.bn_final = nnx.BatchNorm(n_filters, rngs=rngs)
        self.conv_final = nnx.Conv(
            in_features=n_filters,
            out_features=3,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, z):
        x = self.dense(z)  # [B, 128] -> [B, 4*4*256]
        x = x.reshape((-1, 4, 4, 256))  # [B, 4*4*256] -> [B, 4, 4, 256] (NHWC)
        x = self.resblock1(x)  # [B, 4, 4, 256] -> [B, 8, 8, n_filters]
        x = self.resblock2(x)  # [B, 8, 8, n_filters] -> [B, 16, 16, n_filters]
        x = self.resblock3(x)  # [B, 16, 16, n_filters] -> [B, 32, 32, n_filters]
        x = self.bn_final(x)
        x = nnx.relu(x)
        x = self.conv_final(x)  # [B, 32, 32, n_filters] -> [B, 32, 32, 3]
        x = nnx.tanh(x)
        return x


class Critic(nnx.Module):
    def __init__(self, n_filters: int = 128, apply_sn: bool = False, *, rngs: nnx.Rngs):
        self.n_filters = n_filters
        self.apply_sn = apply_sn

        self.resblock1 = ResBlockDown(3, n_filters, rngs=rngs, use_sn=apply_sn)
        self.resblock2 = ResBlockDown(n_filters, n_filters, rngs=rngs, use_sn=apply_sn)
        self.resblock3 = ResBlockDown(n_filters, n_filters, rngs=rngs, use_sn=apply_sn)
        self.resblock4 = ResBlockDown(n_filters, n_filters, rngs=rngs, use_sn=apply_sn)

        if apply_sn:
            from utils.spectral_norm import SpectralNormLinear

            self.dense = SpectralNormLinear(1, rngs=rngs)
        else:
            self.dense = nnx.Linear(in_features=n_filters, out_features=1, rngs=rngs)

    def __call__(self, x):
        x = self.resblock1(x)  # [B, 32, 32, 3] -> [B, 16, 16, n_filters]
        x = self.resblock2(x)  # [B, 16, 16, n_filters] -> [B, 8, 8, n_filters]
        x = self.resblock3(x)  # [B, 8, 8, n_filters] -> [B, 4, 4, n_filters]
        x = self.resblock4(x)  # [B, 4, 4, n_filters] -> [B, 2, 2, n_filters]
        x = nnx.relu(x)
        x = jnp.sum(x, axis=(1, 2))  # global sum pooling over H and W
        x = self.dense(x)  # [B, n_filters] -> [B, 1]
        return x.squeeze(-1)
