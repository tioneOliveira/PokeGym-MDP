import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
import csv
import matplotlib.pyplot as plt
from PokeLight import PokeLightEnv

class MCAgent:
    def __init__(self, env, max_hp, epsilon=0.1, gamma=1.0):
        self.env = env
        self.max_hp = max_hp
        self.nS = (max_hp + 1) ** 2 * 6 * 6
        self.nA = env.action_space.n

        # Q(s,a) inicializado para 0
        self.Q = np.zeros((self.nS, self.nA))

        # Política inicial ε-greedy
        self.policy = np.ones((self.nS, self.nA)) / self.nA

        # Contadores para médias incrementais
        self.returns_count = np.zeros((self.nS, self.nA))

        self.epsilon = epsilon
        self.gamma = gamma

        # Logs de métricas
        self.rewards_log = []
        self.wins_log = []

    def state_to_index(self, hp_a, hp_o, tipo_a, tipo_o):
        # transforma os dados do estado em um indice
        return ((hp_a * (self.max_hp + 1) + hp_o) * 6 + tipo_a) * 6 + tipo_o

    def generate_episode(self):
        obs, info = self.env.reset()
        episode = []
        total_reward = 0
        win = 0

        while True:
            tipo_a, tipo_o, hp_a, hp_o = obs
            s = self.state_to_index(hp_a, hp_o, tipo_a, tipo_o)

            # Escolhe ação y-greedy
            if np.random.rand() < self.epsilon:
                a = np.random.choice(self.nA)
            else:
                a = np.argmax(self.Q[s])

            # executa ação
            obs, reward, done, trunc, info = self.env.step(a)
            episode.append((s, a, reward))
            total_reward += reward

            if done or trunc:
                tipo_a_new, tipo_o_new, hp_a_new, hp_o_new = obs
                # contabiliza vitoria para plot de metricas
                if hp_o_new == 0:
                    win = 1
                else:
                    win = 0
                break

        return episode, total_reward, win

    def update_Q(self, episode):
        G = 0
        for (s, a, r) in reversed(episode):
            G = self.gamma * G + r
            self.returns_count[s, a] += 1
            self.Q[s, a] += (G - self.Q[s, a]) / self.returns_count[s, a]

    def improve_policy(self):
        for s in range(self.nS):
            best_a = np.argmax(self.Q[s])
            self.policy[s] = self.epsilon / self.nA
            self.policy[s, best_a] += 1 - self.epsilon

    def train(self, num_episodes=1000):
        for ep in range(num_episodes):
            episode, total_reward, win = self.generate_episode()
            self.update_Q(episode)
            self.improve_policy()

            # log de métricas
            self.rewards_log.append(total_reward)
            self.wins_log.append(win)

            if (ep + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_log[-num_episodes:])
                win_rate = np.mean(self.wins_log[-num_episodes:])
                print(f"Episódio {ep+1}: Recompensa média={avg_reward:.2f}, Taxa de vitória={win_rate:.2f}")

    def run_policy(self, max_steps=100, delay=0.5):
        obs, info = self.env.reset()
        accumulated_reward = 0

        for _ in range(max_steps):
            tipo_a, tipo_o, hp_a, hp_o = obs
            s = self.state_to_index(hp_a, hp_o, tipo_a, tipo_o)

            a = np.argmax(self.policy[s])
            obs, reward, done, trunc, info = self.env.step(a)
            accumulated_reward += reward

            self.env.render()
            time.sleep(delay)

            if done or trunc:
                break

        return accumulated_reward

    def export_policy_to_csv(self, filename="mc_policy.csv"):
        # Exporta a politica para um CSV, estava curioso.
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Blastoise", "Charizard", "Venosaur", "Machamp", "Gengar", "Mewtwo"])
            for s in range(self.nS):
                writer.writerow(self.policy[s])
        print(f"Política MC exportada para {filename}")

    def plot_metrics(self):
        # média acumulada winrate
        wins_cumavg = np.cumsum(self.wins_log) / np.arange(1, len(self.wins_log) + 1)

        plt.figure(figsize=(8,5))

        plt.plot(wins_cumavg)
        plt.title("Taxa de Vitória Acumulada")
        plt.xlabel("Episódios")
        plt.ylabel("Taxa de Vitória")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    max_hp = 20

    # treino sem render
    env = PokeLightEnv(render_mode=None, max_hp=max_hp, fps=30)
    mc_agent = MCAgent(env, max_hp=max_hp, epsilon=0.1, gamma=1.0)
    mc_agent.train(num_episodes=4000)
    mc_agent.export_policy_to_csv("mc_policy.csv")
    env.close()

    '''
    # plot métricas
    mc_agent.plot_metrics()
    '''

    # teste com render
    env = PokeLightEnv(render_mode="human", max_hp=max_hp, fps=30)
    mc_agent.env = env
    acc_reward = mc_agent.run_policy()
    print("Recompensa acumulada (MC):", acc_reward)
    env.close()