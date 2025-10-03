import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from PokeLightEnv import PokeLightEnv

class VIAgente:
    def __init__(self, env, max_hp):
        self.env = env
        self.max_hp = max_hp
        self.nS = (max_hp + 1) ** 2 * 6 * 6 
        self.nA = env.action_space.n
        self.V = np.zeros(self.nS)
        self.policy = np.zeros((self.nS, self.nA))

    def state_to_index(self, hp_a, hp_o, tipo_a, tipo_o):
        return ((hp_a * (self.max_hp + 1) + hp_o) * 6 + tipo_a) * 6 + tipo_o

    def index_to_state(self, idx):
        tipo_o = idx % 6
        idx //= 6
        tipo_a = idx % 6
        idx //= 6
        hp_o = idx % (self.max_hp + 1)
        idx //= (self.max_hp + 1)
        hp_a = idx
        return hp_a, hp_o, tipo_a, tipo_o

    def one_step_lookahead(self, state_idx, discount_factor=1.0):
        A = np.zeros(self.nA)
        hp_a, hp_o, tipo_a, tipo_o = self.index_to_state(state_idx)

        for a in range(self.nA):
            dano_agente, _ = self.env.calcular_dano(a, tipo_o)
            hp_o_novo = max(hp_o - dano_agente, 0)

            # recompensa imediata
            if dano_agente == 4:
                reward = 2
            elif dano_agente == 2:
                reward = -5
            else:
                reward = 0

            # estado terminal: oponente zerou HP
            if hp_o_novo == 0:
                A[a] = reward + 10
                continue

            dano_oponente, _ = self.env.calcular_dano(tipo_o, a)
            hp_a_novo = max(hp_a - dano_oponente, 0)

            # estado terminal: agente zerou HP
            if hp_a_novo == 0:
                A[a] = reward - 30
                continue

            # soma valor futuro esperado (simula ações do oponente)
            expected_V = 0
            for next_p in range(self.nA):
                idx_novo = self.state_to_index(hp_a_novo, hp_o_novo, a, next_p)
                expected_V += self.V[idx_novo]
            expected_V /= self.nA

            A[a] = reward + discount_factor * expected_V

        return A

    def run_policy(self, max_steps=100):
        obs, info = self.env.reset()
        accumulated_reward = 0
        for _ in range(max_steps):
            tipo_a, tipo_o, hp_a, hp_o = obs
            state_idx = self.state_to_index(hp_a, hp_o, tipo_a, tipo_o)
            action = np.argmax(self.policy[state_idx])
            obs, reward, done, trunc, info = self.env.step(action)
            accumulated_reward += reward
            if done or trunc:
                break
        return accumulated_reward

    def run_value_iteration(self, theta=1e-6, discount_factor=1.0):
        while True:
            delta = 0
            for s in range(self.nS):
                v = self.V[s]
                A = self.one_step_lookahead(s, discount_factor)
                self.V[s] = np.max(A)
                delta = max(delta, abs(v - self.V[s]))
            if delta < theta:
                break

        # extrair política determinística
        for s in range(self.nS):
            A = self.one_step_lookahead(s, discount_factor)
            best_action = np.argmax(A)
            self.policy[s] = np.eye(self.nA)[best_action]

        return self.policy, self.V

    # testar hiperparâmetros
    def testar_hiperparametros(self, discount_factors, thetas, max_hp=20, max_steps=100):
        results = []

        for discount_factor in discount_factors:
            for theta in thetas:
                agent = VIAgente(self.env, max_hp=max_hp)
                start_time = time.perf_counter()
                policy, V = agent.run_value_iteration(theta=theta, discount_factor=discount_factor)
                end_time = time.perf_counter()
                exec_time = end_time - start_time
                reward = agent.run_policy(max_steps=max_steps)
                results.append((discount_factor, theta, exec_time, np.mean(V), reward))
                print(f"Discount Factor: {discount_factor}, Theta: {theta}, "
                      f"Execution Time: {exec_time:.2f}s, Mean V: {np.mean(V):.2f}, Reward: {reward}")

        # Plotar resultados por theta
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for discount_factor in discount_factors:
            data = [r for r in results if r[0] == discount_factor]
            axs[0].plot([r[1] for r in data], [r[2] for r in data], label=f"γ={discount_factor}")
            axs[1].plot([r[1] for r in data], [r[4] for r in data], label=f"γ={discount_factor}")
            axs[2].plot([r[1] for r in data], [r[3] for r in data], label=f"γ={discount_factor}")

        axs[0].set_title("Tempo de execução por θ")
        axs[1].set_title("Recompensa média por θ")
        axs[2].set_title("Média da função V por θ")
        for ax in axs:
            ax.set_xlabel("θ (theta)")
            ax.legend()
        plt.show()

        # Plotar resultados por discount factor
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for theta in thetas:
            data = [r for r in results if r[1] == theta]
            axs[0].plot([r[0] for r in data], [r[2] for r in data], label=f"θ={theta}")
            axs[1].plot([r[0] for r in data], [r[4] for r in data], label=f"θ={theta}")
            axs[2].plot([r[0] for r in data], [r[3] for r in data], label=f"θ={theta}")

        axs[0].set_title("Tempo de execução por γ")
        axs[1].set_title("Recompensa média por γ")
        axs[2].set_title("Média da função V por γ")
        for ax in axs:
            ax.set_xlabel("γ (discount factor)")
            ax.legend()
        plt.show()

    # exportar política
    def export_policy_to_csv(self, filename="policy.csv"):
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Blastoise", "Charizard", "Venosaur", "Machamp", "Gengar", "Mewtwo"])
            for s in range(self.nS):
                writer.writerow(self.policy[s])
        print(f"Policy exportada para {filename}")


if __name__ == "__main__":
    max_hp = 20
    env = PokeLightEnv(render_mode=None, max_hp=max_hp, fps=30)

    vi_agent = VIAgente(env, max_hp=max_hp)

    discount_factors = [0.5, 0.7, 0.9, 0.99]
    thetas = [1e-2, 1e-4, 1e-6]

    vi_agent.testar_hiperparametros(discount_factors, thetas, max_hp=max_hp, max_steps=100)
