import argparse
import os
import secrets

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
import dm_pix as pix
import numpy as np
from flax import nnx
from tqdm import tqdm

from data_loader import get_data_loader, cifar10_preprocess, celeba_preprocess
from model import FlowMatchingModel
from utils import load_yaml, save_json



@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, x1: jnp.ndarray, rngs: nnx.Rngs):
    """Standard Flow Matching training step."""
    aug_key = rngs.params()
    x1 = pix.random_brightness(key=aug_key, image=x1, max_delta=0.1)
    x1 = pix.random_flip_left_right(key=aug_key, image=x1)

    def loss_fn(model):
        x0 = jax.random.normal(rngs.params(), x1.shape)
        t = jax.random.uniform(rngs.params(), (x1.shape[0],))
        t_expanded = t[:, None, None, None]

        xt = (1.0 - t_expanded) * x0 + t_expanded * x1
        ut = x1 - x0
        vt = model(xt, t)
        return jnp.mean((vt - ut) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return {"loss": loss}


@nnx.jit
def eval_step(model: nnx.Module, x1: jnp.ndarray, rngs: nnx.Rngs):
    """Deterministic evaluation step."""
    x0 = jax.random.normal(rngs.params(), x1.shape)
    t = jax.random.uniform(rngs.params(), (x1.shape[0],))
    t_expanded = t[:, None, None, None]

    xt = (1.0 - t_expanded) * x0 + t_expanded * x1
    ut = x1 - x0
    vt = model(xt, t)
    loss = jnp.mean((vt - ut) ** 2)
    return {"loss": loss}


def sample(model: nnx.Module, rng: jax.Array, num_samples=8, res=(32, 32, 3)):
    """Inference/Sampling via ODE integration."""
    x = jax.random.normal(rng, (num_samples, *res))
    steps = 100
    dt = 1.0 / steps
    for i in range(steps):
        t = jnp.ones((num_samples,)) * (i / steps)
        v = model(x, t)
        x = x + dt * v
    return x




class Experiment:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.seed = args.seed
        self.dataset_key = "cifar10" if "cifar10" in args.dataset else "celeba"

        self.rngs = nnx.Rngs(self.seed)

        self.model = FlowMatchingModel(
            channels=3,
            base_dim=config[f"{self.dataset_key}_base_dim"],
            rngs=self.rngs,
        )

        self.schedule = optax.cosine_decay_schedule(
            config[f"{self.dataset_key}_learning_rate"],
            config[f"{self.dataset_key}_decay_steps"],
            alpha=config[f"{self.dataset_key}_alpha"],
        )
        self.optimizer = nnx.Optimizer(self.model, optax.adam(self.schedule))

        self.mngr = ocp.CheckpointManager(
            os.path.abspath(f"checkpoints/{self.dataset_key}_seed_{self.seed}"),
            ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
            options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True),
        )

    def run_eval(self, val_it, step):
        losses = []
        for _ in range(20):
            batch = next(val_it)
            metrics = eval_step(self.model, batch["image"], self.rngs)
            losses.append(metrics["loss"])

        avg_loss = float(np.mean(losses))

        img_size = self.config[f"{self.dataset_key}_img_size"]
        vis_samples = sample(self.model, jax.random.key(step), res=(img_size, img_size, 3))
        vis_samples = np.array(jnp.clip((vis_samples + 1.0) * 127.5, 0, 255).astype(jnp.uint8))

        return {"eval/loss_avg": avg_loss, "samples": [wandb.Image(s) for s in vis_samples]}

    def train(self):
        train_it = get_data_loader(
            dataset_path=self.args.dataset,
            batch_size=self.config[f"{self.dataset_key}_batch_size"],
            seed=self.seed,
            split="train[:90%]",
            preprocess_fn=cifar10_preprocess
            if "cifar10" in self.args.dataset
            else celeba_preprocess,
            target_size=self.config[f"{self.dataset_key}_img_size"],
            repeat=True,
        )
        val_it = get_data_loader(
            dataset_path=self.args.dataset,
            batch_size=self.config[f"{self.dataset_key}_batch_size"],
            seed=self.seed + 1,
            split="train[90%:]",
            preprocess_fn=cifar10_preprocess
            if "cifar10" in self.args.dataset
            else celeba_preprocess,
            target_size=self.config[f"{self.dataset_key}_img_size"],
            repeat=True,  # Repeating val is okay IF you control the loop length
        )

        training_steps = self.config[f"{self.dataset_key}_training_steps"]
        eval_every = self.config.get(f"{self.dataset_key}_eval_every", 1000)

        history = {"step": [], "train/loss_avg": [], "eval/loss_avg": []}
        running_train_loss = []

        os.makedirs("../results", exist_ok=True)

        pbar = tqdm(range(training_steps), desc=f"Training {self.dataset_key}")
        for step in pbar:
            batch = next(train_it)
            metrics = train_step(self.model, self.optimizer, batch["image"], self.rngs)

            loss_val = float(metrics["loss"])
            running_train_loss.append(loss_val)

            if step % 100 == 0:
                wandb.log({"train/loss_step": loss_val}, step=step)
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            if step > 0 and step % eval_every == 0:
                avg_train_loss = np.mean(running_train_loss)
                eval_results = self.run_eval(val_it, step)  # WandB eval/loss_avg logged inside

                history["step"].append(int(step))
                history["train/loss_avg"].append(float(avg_train_loss))
                history["eval/loss_avg"].append(float(eval_results["eval/loss_avg"]))

                wandb.log(
                    {
                        "train/loss_avg": avg_train_loss,
                        "eval/loss_avg": eval_results["eval/loss_avg"],
                        "eval/samples": eval_results["samples"],
                    },
                    step=step,
                )

                running_train_loss = []
                save_json(history, f"../results/history_{self.dataset_key}_seed_{self.seed}.json")
                self.mngr.save(step, nnx.state(self.optimizer))

                current_lr = self.schedule(step)
                print(
                    f"\n[Step {step}] Train Avg: {avg_train_loss:.4f} | Val Avg: {eval_results['eval/loss_avg']:.4f} | LR: {current_lr:.6f}"
                )

        save_json(history, f"../results/history_{self.dataset_key}_seed_{self.seed}_final.json")

        os.makedirs("models", exist_ok=True)
        final_mngr = ocp.CheckpointManager(
            os.path.abspath(f"models/{self.dataset_key}_model_seed_{self.seed}"),
            ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        )
        final_mngr.save(0, nnx.state(self.model))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["uoft-cs/cifar10", "flwrlabs/celeba"],
        default="uoft-cs/cifar10",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=secrets.randbelow(2**32),
    )
    args = parser.parse_args()

    config = load_yaml("hyperparameters.yaml")
    

    exp = Experiment(args, config)
    
    run_name = f"{args.dataset.split('/')[-1]}_dim{config[f'{exp.dataset_key}_base_dim']}_s{args.seed}"
    wandb.init(
        project="Flow-Matching", 
        name=run_name,
        config=config
    )
    
    exp.train()


if __name__ == "__main__":
    main()
