import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PokeLightEnv import PokeLightEnv

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        # compartilhado
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # ator
        self.policy_logits = nn.Linear(hidden_size, action_dim)
        # crítico
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_logits(x), self.value(x).squeeze(-1)

# utilidades

def compute_returns_and_advantages(rewards, dones, values, last_value, gamma=0.99, gae_lambda=1.0):
    # calculo do retorno descontado
    returns = np.zeros_like(rewards)
    running_return = last_value
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
        returns[t] = running_return
    # calculo das vantagens
    advantages = returns - values
    return returns, advantages

# treino A2C
def train_a2c(env, model, optimizer, device=torch.device("cpu"),
              total_steps=200_000, n_steps=6, gamma=0.99,
              value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
              log_interval=2000, gae_lambda=1.0):

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    # inicializa em 0
    step = 0
    episode_rewards = deque(maxlen=100)
    current_episode_reward = 0.0
    episode_count = 0

    # timer para logs
    start_time = time.time()

    # loop principal de coleta de trajetorias e treinamento
    while step < total_steps:
        # armazenamento
        obs_batch = []
        actions_batch = []
        rewards_batch = []
        dones_batch = []
        values_batch = []
        log_probs_batch = []

        # coleta de interações com o ambiente
        for _ in range(n_steps):
            logits, value = model(obs.unsqueeze(0))  # adiciona dimensão de batch
            logits = logits.squeeze(0)
            value = value.squeeze(0)

            # distribuição categórica para ações discretas
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            # armazena informações da transição
            obs_batch.append(obs.cpu().numpy())
            actions_batch.append(int(action.item()))
            values_batch.append(value.item())
            log_probs_batch.append(dist.log_prob(action).item())

            # passo no ambiente
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = bool(terminated or truncated)

            rewards_batch.append(float(reward))
            dones_batch.append(float(done))

            # recompensa acumulada do episódio
            current_episode_reward += float(reward)

            step += 1

            if done:
                # episódio finalizado
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                episode_count += 1
                next_obs, _ = env.reset()

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # finaliza caso ultrapasse total_steps
            if step >= total_steps:
                break

        # valor de bootstrap do último estado
        with torch.no_grad():
            _, last_value = model(obs.unsqueeze(0))
            last_value = last_value.squeeze(0).cpu().numpy()

        # conversão para numpy para o cálculo das vantagens
        rewards_arr = np.array(rewards_batch, dtype=np.float32)
        dones_arr = np.array(dones_batch, dtype=np.float32)
        values_arr = np.array(values_batch, dtype=np.float32)

        returns, advantages = compute_returns_and_advantages(
            rewards_arr, dones_arr, values_arr, last_value, gamma=gamma
        )

        # conversão dos dados coletados para tensores PyTorch
        obs_tensor = torch.tensor(np.array(obs_batch, dtype=np.float32), device=device)
        actions_tensor = torch.tensor(np.array(actions_batch), device=device)
        returns_tensor = torch.tensor(returns, device=device, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, device=device, dtype=torch.float32)
        old_log_probs_tensor = torch.tensor(np.array(log_probs_batch, dtype=np.float32), device=device)

        # forward para recomputar políticas e valores
        logits_batch, values_pred = model(obs_tensor)
        dist_batch = torch.distributions.Categorical(logits=logits_batch)
        log_probs = dist_batch.log_prob(actions_tensor)
        entropy = dist_batch.entropy().mean()  # mede exploração

        # perda do crítico
        value_loss = (returns_tensor - values_pred).pow(2).mean()
        # perda do ator
        policy_loss = -(advantages_tensor.detach() * log_probs).mean()

        # perda total com entropia para incentivar exploração
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # logs
        if step % log_interval < n_steps:
            elapsed = time.time() - start_time
            avg_reward = float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else 0.0
            print(f"Step: {step}/{total_steps} | Episodes: {episode_count} | AvgR(100): {avg_reward:.2f} | Loss: {loss.item():.4f} | Crit: {policy_loss.item():.4f} | Ent: {entropy.item():.4f} | Time: {elapsed:.1f}s")

    return model


if __name__ == "__main__":
    # hiperparâmetros do treino
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_steps = 150_000
    n_steps = 6
    lr = 2.5e-4

    # inicialização do ambiente
    env = PokeLightEnv(render_mode=None, max_hp=10, fps=30)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(obs_dim, action_dim, hidden_size=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # treinamento A2C
    trained_model = train_a2c(env, model, optimizer, device=device,
                              total_steps=total_steps, n_steps=n_steps,
                              gamma=0.99, value_coef=0.5, entropy_coef=0.01,
                              max_grad_norm=0.5, log_interval=2000)

    print("Treinamento finalizado")

    env.close()
