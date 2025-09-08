import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        inner_steps,
        inner_lr,
        train_loader,
        loss_fn,
        epochs,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device

    def inner_loop(self, support_x, support_y):
        fast_weights = {
            name: param.clone() for name, param in self.model.named_parameters()
        }

        for _ in range(self.inner_steps):
            logits = self.model.functional_forward(support_x, fast_weights)
            loss = self.loss_fn(logits, support_y)
            grads = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True
            )

            fast_weights = {
                name: w - self.inner_lr * g
                for (name, w), g in zip(fast_weights.items(), grads)
            }

        return fast_weights

    def train(self, file_name):
        self.model.train()
        avg_losses = []

        for epoch in range(self.epochs):
            total_loss = 0
            num_episodes = 0
            for batch in tqdm(self.train_loader):
                support_x = batch["support_images"].to(self.device)
                support_y = batch["support_labels"].to(self.device)
                query_x = batch["query_images"].to(self.device)
                query_y = batch["query_labels"].to(self.device)

                support_x = support_x.view(-1, 1, 28, 28) # [batch_size, n_way*k_shot, in_channel, 28, 28] -> [batch_size*n_way*k_shot, 1, 28, 28]
                support_y = support_y.view(-1) # [batch_size, n_way*k_shot] -> [batch_size*n_way*k_shot]
                query_x = query_x.view(-1, 1, 28, 28) # [batch_size, n_way*q_query, in_channel, 28, 28] -> [batch_size*n_way*q_query, 1, 28, 28]
                query_y = query_y.view(-1) # [batch_size, n_way*q_query] -> [batch_size*n_way*q_query]
                fast_weights = self.inner_loop(support_x, support_y)

                logits_query = self.model.functional_forward(query_x, fast_weights)
              
                loss = self.loss_fn(logits_query, query_y)
                total_loss += loss.item()
                num_episodes += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            avg_loss = total_loss / num_episodes
            avg_losses.append(avg_loss)
            tqdm.write(f"Epoch {epoch + 1}/{self.epochs}, Avg Loss: {avg_loss:.4f}")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"{file_name}.pth",
        )

        return avg_losses
