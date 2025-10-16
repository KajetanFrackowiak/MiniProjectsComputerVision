import jax.numpy as jnp
from flax import nnx


def depth_to_space(x: jnp.ndarray, block_size: int = 2) -> jnp.ndarray:
    """
    Rearranges data from depth into blocks of spatial data.
    Args:
        x: Input tensor of shape (B, H, W, C)
        block_size: Size of the spatial block
    Returns:
        Tensor of shape (B, H*block_size, W*block_size, C/(block_size^2))
    """
    B, H, W, C = x.shape
    r = block_size
    out_c = C // (r * r)
    x = x.reshape(B, H, W, r, r, out_c)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H * r, W * r, out_c)


def space_to_depth(x: jnp.ndarray, block_size: int = 2) -> jnp.ndarray:
    """
    Rearranges blocks of spatial data into depth.
    Args:
        x: Input tensor of shape (B, H, W, C)
        block_size: Size of the spatial block
    Returns:
        Tensor of shape (B, H/block_size, W/block_size, C*(block_size^2))
    """
    B, H, W, C = x.shape
    r = block_size
    out_c = C * (r * r)
    x = x.reshape(B, H // r, r, W // r, r, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H // r, W // r, out_c)



class ResBlockUp(nnx.Module):
    """
    Residual block with upsampling
    Args:
        input_dim: number of input channels
        output_dim: number of output channels
        rngs: random number generators for parameter initialization
    Methods:
        __call__(x): forward pass
    Returns:
        output tensor of shape (B, H*2, W*2, output_dim)
    """
    def __init__(self, input_dim: int, output_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.conv1 = nnx.Conv(input_dim, output_dim, kernel_size=(3,3), strides=(1,1), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(output_dim, output_dim, kernel_size=(3,3), strides=(1,1), padding="SAME", rngs=rngs)
        self.conv_skip = nnx.Conv(input_dim, output_dim, kernel_size=(1,1), strides=(1,1), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nnx.relu(self.conv1(x))
        x = depth_to_space(x, 2)
        x = nnx.relu(self.conv2(x))
        shortcut = depth_to_space(residual, 2)
        shortcut = self.conv_skip(shortcut)
        return x + shortcut


class ResBlockDown(nnx.Module):
    """
    Residual block with downsampling
    Args:
        input_dim: number of input channels
        output_dim: number of output channels
        rngs: random number generators for parameter initialization
    Methods:
        __call__(x): forward pass
    Returns:
        output tensor of shape (B, H/2, W/2, output_dim)
    """
    def __init__(self, input_dim: int, output_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.conv1 = nnx.Conv(input_dim, output_dim, kernel_size=(3,3), strides=(1,1), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(output_dim, output_dim, kernel_size=(3,3), strides=(1,1), padding="SAME", rngs=rngs)
        self.conv_skip = nnx.Conv(input_dim, output_dim, kernel_size=(1,1), strides=(1,1), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nnx.relu(self.conv1(x))
        x = nnx.avg_pool(nnx.relu(self.conv2(x)), window_shape=(2,2), strides=(2,2), padding="SAME")
        shortcut = nnx.avg_pool(residual, window_shape=(2,2), strides=(2,2), padding="SAME")
        shortcut = self.conv_skip(shortcut)
        return x + shortcut


class Encoder(nnx.Module):
    """
    Encoder network
    Args:
        input_dim: number of input channels
        base_dim: base number of filters
        rngs: random number generators for parameter initialization
    Methods:
        __call__(x): forward pass
    Returns:
        output tensor of shape (B, H/8, W/8, base_dim*4)
    """
    def __init__(self, input_dim: int, base_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.conv_initial = nnx.Conv(input_dim, base_dim, kernel_size=(3,3), strides=(1,1), padding="SAME", rngs=rngs)
        self.res1 = ResBlockDown(base_dim, base_dim, rngs)
        self.res2 = ResBlockDown(base_dim, base_dim*2, rngs)
        self.res3 = ResBlockDown(base_dim*2, base_dim*4, rngs)
        self.conv_final = nnx.Conv(base_dim*4, base_dim*4, kernel_size=(3,3), strides=(1,1), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv_initial(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.conv_final(x)


class Codebook(nnx.Module):
    """
    Vector Quantization Codebook
    Args:
        codebook_size: number of embeddings in the codebook
        embedding_dim: dimension of each embedding
        rngs: random number generators for parameter initialization
    Methods:
        __call__(z): forward pass
    Returns:
        quantized tensor of shape (B, H, W, D) and indices of shape (B, H, W)
    """
    def __init__(self, codebook_size: int, embedding_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.embeddings = self.param("embeddings", nnx.initializers.normal(0.02), (codebook_size, embedding_dim), rngs=rngs) # [K, D]

    def __call__(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        z_flat = z.reshape(-1, z.shape[-1])
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*<z,e>, shape: (N, K)
        distances = jnp.sum(z_flat**2, 1, keepdims=True) + jnp.sum(self.embeddings**2,1) - 2 * jnp.dot(z_flat, self.embeddings.T)
        indices = jnp.argmin(distances, axis=1) # (N,)
        quantized = self.embeddings[indices].reshape(z.shape) # (N, D) -> (B, H, W, D)
        indices = indices.reshape(z.shape[0], z.shape[1], z.shape[2])
        return quantized, indices


class Generator(nnx.Module):
    """
    Generator network
    Args:
        latent_dim: dimension of the latent vector
        n_filters: number of filters in the first layer
        rngs: random number generators for parameter initialization
    Methods:
        __call__(z): forward pass
    Returns:
        output tensor of shape (B, H, W, 3)
    """
    def __init__(self, latent_dim: int, n_filters: int, rngs: nnx.Rngs):
        super().__init__()
        self.dense = nnx.Linear(latent_dim, n_filters, rngs=rngs)
        self.res1 = ResBlockUp(n_filters, n_filters, rngs)
        self.res2 = ResBlockUp(n_filters, n_filters, rngs)
        self.res3 = ResBlockUp(n_filters, n_filters, rngs)
        self.conv_final = nnx.Conv(n_filters, 3, kernel_size=(3,3), strides=(1,1), padding="SAME", rngs=rngs)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = self.dense(z)
        z = z.reshape(z.shape[0], 4, 4, -1)
        z = self.res1(z)
        z = self.res2(z)
        z = self.res3(z)
        return self.conv_final(z)

class Discriminator(nnx.Module):
    """
    Discriminator network
    Args:
        input_dim: number of input channels
        base_dim: base number of filters
        rngs: random number generators for parameter initialization
    Methods:
        __call__(x): forward pass
    Returns:
        output tensor of shape (B, 1)
    """
    def __init__(self, input_dim: int, base_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.conv1 = nnx.Conv(input_dim, base_dim, kernel_size=(3,3), strides=(1,1), padding="SAME", rngs=rngs)
        self.res1 = ResBlockDown(base_dim, base_dim, rngs)
        self.res2 = ResBlockDown(base_dim, base_dim*2, rngs)
        self.res3 = ResBlockDown(base_dim*2, base_dim*4, rngs)
        self.dense_out = nnx.Linear(base_dim*4, 1, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = jnp.mean(x, axis=(1,2)) # from (B, H, W, C) to (B, C) global pooling
        return self.dense_out(x)


class VQGAN(nnx.Module):
    """
    VQ-GAN model
    Args:
        input_dim: number of input channels
        base_dim: base number of filters
        codebook_size: number of embeddings in the codebook
        embedding_dim: dimension of each embedding
        latent_dim: dimension of the latent vector
        rngs: random number generators for parameter initialization
    Methods:
        __call__(x): forward pass
    Returns:
        reconstructed tensor of shape (B, H, W, 3), quantized tensor of shape (B, H', W', D), discriminator logits of shape (B, 1)
    """
    def __init__(self, input_dim, base_dim, codebook_size, embedding_dim, latent_dim, rngs: nnx.Rngs):
        super().__init__()
        self.encoder = Encoder(input_dim, base_dim, rngs)
        self.codebook = Codebook(codebook_size, embedding_dim, rngs)
        self.generator = Generator(latent_dim, embedding_dim, rngs)
        self.discriminator = Discriminator(input_dim, base_dim, rngs)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_e = self.encoder(x)  # [B, H', W', D]
        z_q, indices = self.codebook(z_e)
        x_hat = self.generator(z_q)
        logits_disc = self.discriminator(x_hat)
        return x_hat, z_q, logits_disc
