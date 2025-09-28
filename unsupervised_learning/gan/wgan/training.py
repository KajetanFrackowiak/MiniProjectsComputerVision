import tensorflow as tf
import wandb


class Trainer:
    def __init__(
        self,
        train_dataset,
        generator,
        critic,
        gen_optimizer,
        crit_optimizer,
        crit_update_interval,
        checkpoint_interval,
        checkpoint_dir,
    ):
        self.train_dataset = train_dataset
        self.generator = generator
        self.critic = critic
        self.gen_optimizer = gen_optimizer
        self.crit_optimizer = crit_optimizer
        self.crit_update_interval = crit_update_interval

    @tf.function
    def critic_train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        z = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as tape:
            fake_images = self.generator(z, training=True)
            real_output = self.critic(real_images, training=True)
            fake_output = self.critic(fake_images, training=True)
            loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

        grads = tape(loss, self.critic.trainable_variables)
        self.crit_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        for var in self.critic.trainable_variables:
            var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value))

        return loss

    @tf.function
    def generator_train_step(self):
        batch_size = self.batch_size
        z = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as tape:
            fake_images = self.generator(z, training=True)
            fake_output = self.critic(fake_images, training=True)
            loss = -tf.reduce_mean(fake_output)

        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

    def train(self):
        training_losses = {"crit_losses": [], "gen_losses": []}

        for epoch in range(self.epochs):
            epoch_losses = {"crit_losses": [], "gen_losses": []}

            for real_images, _ in self.train_dataset:
                real_images = tf.reshape(real_images, [tf.shape(real_images)[0], -1])

                for _ in range(self.n_critic):
                    crit_loss = self.critic_train_step(real_images)

                gen_loss = self.generator_train_step()

                epoch_losses["crit_losses"].append(crit_loss)
                epoch_losses["gen_losses"].append(gen_loss)

            avg_crit_loss = tf.reduce_mean(epoch_losses["crit_losses"])
            avg_gen_loss = tf.reduce_mean(epoch_losses["gen_losses"])

            training_losses["crit_losses"].append(avg_gen_loss)
            training_losses["gen_losses"].append(avg_gen_loss)

            wandb.log(
                {"critic_avg_loss": avg_crit_loss, "generator_avg_loss": avg_gen_loss}
            )
            print(
                f"Epoch {epoch + 1}/{self.epochs} | ",
                f"Critic Loss: {avg_crit_loss} | ",
                f"Generator Loss: {avg_gen_loss}",
            )
            