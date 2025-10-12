import jax
import jax.numpy as jnp


# Wasserstein GAN loss
def wgan_loss(D_real, D_fake):
    # L_D = E[D(x)] - E[D(G(z))], L_G = -E[D(G(z))]
    d_loss = jnp.mean(D_fake) - jnp.mean(D_real)
    g_loss = -jnp.mean(D_fake)
    return d_loss, g_loss


# Gradient Penalty for WGAN-GP
def gradient_penalty(D_model, real, fake, rng):
    # L_GP = E[(||∇D(x) ||_2 - 1)^2]
    alpha = jax.random.uniform(rng, (real.shape[0], 1, 1, 1))
    interpolated = alpha * real + (1 - alpha) * fake

    def grad_fn(x):
        return jnp.mean(D_model(x))

    grads = jax.grad(grad_fn)(interpolated)  # Gradient w.r.t. interpolated samples
    norm = jnp.sqrt(jnp.sum(grads**2, axis=[1, 2, 3]))  # Compute L2 norm per sample
    penalty = jnp.mean((norm - 1.0) ** 2)  # L2 norm penalty
    return penalty


# WGAN-GP loss
def wgan_gp_loss(D_real, D_fake, gp, lambda_gp=10.0):
    # L_D = E[D(x)] - E[D(G(z))] + λ * GP, L_G = -E[D(G(z))]
    d_loss = jnp.mean(D_fake) - jnp.mean(D_real) + lambda_gp * gp
    g_loss = -jnp.mean(D_fake)
    return d_loss, g_loss
