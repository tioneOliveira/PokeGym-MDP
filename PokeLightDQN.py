import random
import collections
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from PokeLightEnv import PokeLightEnv

Transition = collections.namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super(DuelingQNetwork, self).__init__()
        # shared
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.shared = nn.Sequential(*layers)

        # value stream
        self.value_head = nn.Sequential(nn.Linear(last, 128), nn.ReLU(), nn.Linear(128, 1))
        # advantage stream
        self.adv_head = nn.Sequential(nn.Linear(last, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        x = self.shared(x)
        v = self.value_head(x)
        a = self.adv_head(x)
        # combine: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class DQNAgente:
    def __init__(
        self,
        env: PokeLightEnv,
        max_hp: int,
        device=None,
        buffer_capacity=200000,
        batch_size=128,
        gamma=0.99,
        lr=2e-4,
        target_update=5000,
        train_start=4000,
    ):
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.max_hp = max_hp

        self.policy_net = DuelingQNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target_net = DuelingQNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_capacity)

        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.train_start = train_start
        self.steps_done = 0

    # helper para extrair ações válidas do obs
    def get_valid_actions_from_obs(self, obs: np.ndarray):
        # obs layout: 0-5 one-hot tipo_agente, 6-11 one-hot tipo_oponente,
        # 12-17 hp_agente, 18-23 hp_oponente
        agent_onehot = obs[0:6]
        agent_idx = int(np.argmax(agent_onehot))
        agent_hps = obs[12:18] * self.max_hp
        valid_actions = []
        # ataque possível se hp do poke ativo > 0
        if agent_hps[agent_idx] > 0.0:
            valid_actions.append(0)
        
        for idx in range(6):
            if agent_hps[idx] > 0 and idx != agent_idx:
                valid_actions.append(idx + 1)
        return valid_actions

    def select_action(self, state: np.ndarray, epsilon: float, valid_actions=None):
        # se nenhuma ação é permitida
        if valid_actions is not None and len(valid_actions) == 0:
            return None

        if valid_actions is None:
            valid_actions = self.get_valid_actions_from_obs(state)

        # exploração
        if random.random() < epsilon:
            return random.choice(valid_actions)

        # greedy
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.policy_net(state_t).cpu().numpy().flatten()

        # mask para ações inválidas
        mask = np.full_like(qvals, -1e9)
        mask[valid_actions] = qvals[valid_actions]

        return int(np.argmax(mask))

    def optimize_model(self):
        if len(self.replay) < self.batch_size:
            return 0.0

        transitions = self.replay.sample(self.batch_size)
        state_batch = torch.tensor(np.vstack(transitions.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(transitions.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.vstack(transitions.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(transitions.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q = reward_batch + (1.0 - done_batch) * (self.gamma * next_q_values)

        loss = nn.functional.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        return float(loss.item())

    def preprocess_obs(self, obs):
        return np.array(obs, dtype=np.float32)

    def _extract_hp(self, state):
        # hps normalizados
        state = np.array(state, dtype=np.float32)
        agent_hp = state[12:18] * self.max_hp
        opp_hp = state[18:24] * self.max_hp
        return agent_hp, opp_hp

    # hiperparametros
    def train(
        self,
        num_episodes=2000,
        max_steps_per_episode=200,
        epsilon_start=1.0,
        epsilon_final=0.02,
        epsilon_decay=150000,
        report_every=50,
    ):
        epsilon = epsilon_start
        losses = []
        episode_rewards = []
        total_steps = 0

        for ep in range(1, num_episodes + 1):
            obs, info = self.env.reset()
            state = self.preprocess_obs(obs)
            valid_actions = self.get_valid_actions_from_obs(state)
            ep_reward = 0

            for t in range(max_steps_per_episode):
                action = self.select_action(state, epsilon, valid_actions)

                # se nenhuma ação é possível, finaliza o episódio
                if action is None:
                    break

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_state = self.preprocess_obs(next_obs)
                next_valid_actions = self.get_valid_actions_from_obs(next_state)

                done = bool(terminated or truncated)

                # Reward shaping usando diferença de HP 
                _, opp_hp_before = self._extract_hp(state)
                _, opp_hp_after = self._extract_hp(next_state)
                delta_opp_hp = float(np.sum(opp_hp_before) - np.sum(opp_hp_after))

                agent_hp_before, _ = self._extract_hp(state)
                agent_hp_after, _ = self._extract_hp(next_state)
                delta_agent_hp = float(np.sum(agent_hp_before) - np.sum(agent_hp_after))

                # escala do shaping
                shaped = 0.5 * delta_opp_hp - 0.7 * delta_agent_hp

                # combina
                stored_reward = reward + shaped

                
                self.replay.push(state, action, stored_reward, next_state, float(done))

                state = next_state
                valid_actions = next_valid_actions
                ep_reward += reward

                if len(self.replay) >= self.train_start:
                    losses.append(self.optimize_model())

                # atualização target por passos
                if self.steps_done % self.target_update == 0 and self.steps_done > 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                self.steps_done += 1
                total_steps += 1

                # epsilon decay exponencial por passos
                epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-total_steps / epsilon_decay)

                if done:
                    break

            episode_rewards.append(ep_reward)

            if ep % report_every == 0:
                avg_reward = np.mean(episode_rewards[-report_every:])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                print(
                    f"Episode {ep}\tAvgReward(last {report_every}): {avg_reward:.2f}\t"
                    f"AvgLoss(last100): {avg_loss:.4f}\tEpsilon: {epsilon:.4f}\tSteps: {self.steps_done:.4f}"
                )

        return {
            "rewards": episode_rewards,
            "losses": losses,
        }

    def evaluate(self, n_episodes=20, max_steps_per_episode=200, render=False):
        rewards = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            state = self.preprocess_obs(obs)
            valid_actions = self.get_valid_actions_from_obs(state)

            ep_reward = 0

            for t in range(max_steps_per_episode):
                action = self.select_action(state, 0.0, valid_actions)

                if action is None:
                    break

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_state = self.preprocess_obs(next_obs)
                next_valid_actions = self.get_valid_actions_from_obs(next_state)

                if render and self.env.render_mode == "human":
                    self.env.render()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return rewards

                state = next_state
                valid_actions = next_valid_actions
                ep_reward += reward

                if terminated or truncated:
                    break

            rewards.append(ep_reward)

        return rewards

    def play_one_episode(self, max_steps=300):
        obs, info = self.env.reset()
        state = self.preprocess_obs(obs)
        valid_actions = self.get_valid_actions_from_obs(state)

        ep_reward = 0

        for t in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return ep_reward

            action = self.select_action(state, 0.0, valid_actions)

            if action is None:
                break

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_state = self.preprocess_obs(next_obs)
            valid_actions = self.get_valid_actions_from_obs(next_state)

            if self.env.render_mode == "human":
                self.env.render()

            state = next_state
            ep_reward += reward

            if terminated or truncated:
                break

        return ep_reward


if __name__ == "__main__":
    max_hp = 10
    env = PokeLightEnv(render_mode=None, max_hp=max_hp, fps=30)

    agent = DQNAgente(env, max_hp=max_hp)

    start = time.time()
    result = agent.train(num_episodes=1200, max_steps_per_episode=100, report_every=50)
    end = time.time()
    print(f"Training finished in {end - start:.2f}s")

    eval_rewards = agent.evaluate(n_episodes=50, render=False)
    print(f"Evaluation mean reward over 50 episodes: {np.mean(eval_rewards):.2f}")

    env.close()

    # demonstração visual
    env = PokeLightEnv(render_mode="human", max_hp=max_hp, fps=30)
    agent.env = env
    env.render()

    final_reward = agent.play_one_episode(max_steps=300)
    print(f"Recompensa total do episódio demonstrativo: {final_reward:.2f}")

    env.close()
