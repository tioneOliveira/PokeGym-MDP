import random
import gymnasium as gym
from gym import spaces
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Tipos usados no combate
# Lista de tipos que o agente reconhecerá
used_types = ["Fire", "Water", "Grass", "Flying", "Poison"]

# Dicionário com informações sobre vantagens, desvantagens e imunidades
pokemon_types = {
    "Fire": {"advantages": ["Grass"], "disadvantages": ["Water"], "immunities": []},
    "Water": {"advantages": ["Fire"], "disadvantages": ["Grass"], "immunities": []},
    "Grass": {"advantages": ["Water"], "disadvantages": ["Fire"], "immunities": []},
    "Flying": {"advantages": ["Grass"], "disadvantages": [], "immunities": []},
    "Poison": {"advantages": ["Grass"], "disadvantages": [], "immunities": []}
}

def type_to_onehot(t, type_set=used_types):
    """Converte um tipo em vetor one-hot para ser usado como entrada do agente"""
    vec = [0]*len(type_set)
    if t in type_set:
        vec[type_set.index(t)] = 1
    return vec

# Classes Move e Pokemon
class Move:
    """Classe que representa um movimento de Pokémon"""
    def __init__(self, name="Tackle", mtype="Normal", stat="physical", power=35, accuracy=95, pp=35, priority=0):
        self.name = name
        self.type = mtype
        self.stat = stat
        self.power = power
        self.total_pp = pp
        self.pp = pp
        self.priority = priority

    def reduce_pp(self):
        """Reduz PP do movimento em 1"""
        if self.pp > 0:
            self.pp -= 1

class Pokemon:
    """Classe que representa um Pokémon"""
    def __init__(self, name="Rattata", type1="Normal", type2="monotype",
                 hp=30, attack=56, defense=35, special=25, speed=72, moves=None):
        self.name = name
        self.type1 = type1
        self.type2 = type2
        self.total_hp = hp
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.special = special
        self.speed = speed
        self.moves = moves if moves else [Move() for _ in range(4)]
        self.fainted = False

    def calculate_type_modifier(self, move_type):
        """
        Calcula o modificador de tipo para dano:
        - x2 se super efetivo
        - x0.5 se pouco efetivo
        - x0 se imunidade
        """
        modifier = 1.0
        for t in [self.type1, self.type2] if self.type2 != "monotype" else [self.type1]:
            info = pokemon_types[t]
            if move_type in info["immunities"]:
                return 0.0
            if move_type in info["advantages"]:
                modifier *= 2
            elif move_type in info["disadvantages"]:
                modifier *= 0.5
        return modifier

    def take_damage(self, dmg):
        """Aplica dano ao Pokémon e verifica se desmaiou"""
        self.hp = max(0, self.hp - dmg)
        if self.hp == 0:
            self.fainted = True

    def make_move(self, target, move_idx):
        """
        Executa um ataque contra outro Pokémon:
        - Verifica PP
        - Calcula acerto
        - Aplica dano com modificadores de tipo e STAB
        """
        move = self.moves[move_idx]
        if move.pp <= 0:
            return None, "NoPP"
        move.reduce_pp()
        if random.randint(1, 100) > move.accuracy:
            return 0, "Miss"

        atk_stat = self.attack if move.stat == "physical" else self.special
        def_stat = target.defense if move.stat == "physical" else target.special
        type_mod = target.calculate_type_modifier(move.type)
        stab = 1.5 if move.type in [self.type1, self.type2] else 1.0
        ratio = atk_stat / def_stat
        base_damage = int((((2 * 100 / 5) + 2) * move.power * ratio / 50 + 2) *
                          (0.85 + random.random() * 0.15))
        damage = int(base_damage * type_mod * stab)
        target.take_damage(damage)

        if type_mod > 1:
            return damage, "SuperEffective"
        elif 0 < type_mod < 1:
            return damage, "NotVeryEffective"
        elif type_mod == 0:
            return 0, "NoEffect"
        else:
            return damage, "Normal"

class Team:
    """Classe que representa um time de Pokémon"""
    def __init__(self, pokemons, active_idx=0):
        self.pokemons = pokemons
        self.active = pokemons[active_idx]

    def switch(self, idx):
        """Troca Pokémon ativo"""
        if 0 <= idx < len(self.pokemons) and not self.pokemons[idx].fainted:
            self.active = self.pokemons[idx]

    def lose(self):
        """Verifica se todo o time foi derrotado"""
        return all(p.fainted for p in self.pokemons)

# Ambiente Pokémon
class PokemonEnv(gym.Env):
    """
    Ambiente customizado de Pokémon para aprendizado por reforço.

    Observações:
        - Vetor contínuo com informações de status dos Pokémons e movimentos.
    Ações:
        - 0-3: usar movimentos
        - 4-6: trocar Pokémon
    Recompensa:
        - Baseada em dano causado, dano recebido e efetividade.
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(7)
        per_pokemon_len = 6 + 4 * (5 + len(used_types)) + 2 * len(used_types)
        self.observation_space = spaces.Box(low=0, high=255, shape=(6 * per_pokemon_len,), dtype=np.float32)
        self.total_wins = 0
        self.total_losses = 0

        # Estatísticas
        self.super_effective_count = 0
        self.not_very_effective_count = 0
        self.switch_count = 0
        self.total_damage_dealt = 0
        self.total_damage_received = 0
        self.pokemon_usage = Counter()

        self.reset()

    def reset(self):
        """Reinicia o ambiente e os times de Pokémon"""
        tackle = Move("Tackle", "Normal", "physical", 35, 95, 35)
        ember = Move("Ember", "Fire", "special", 40, 100, 25)
        water_gun = Move("Water Gun", "Water", "special", 40, 100, 25)
        vine_whip = Move("Vine Whip", "Grass", "physical", 45, 100, 25)

        # Times fixos
        self.team1 = Team([
            Pokemon("Charizard", "Fire", "Flying", 78, 84, 78, 85, 100, [ember, tackle, ember, tackle]),
            Pokemon("Blastoise", "Water", "monotype", 79, 83, 100, 85, 78, [water_gun, tackle, water_gun, tackle]),
            Pokemon("Venusaur", "Grass", "Poison", 80, 82, 83, 100, 80, [vine_whip, tackle, vine_whip, tackle])
        ])
        self.team2 = Team([
            Pokemon("Charizard", "Fire", "Flying", 78, 84, 78, 85, 100, [ember, tackle, ember, tackle]),
            Pokemon("Blastoise", "Water", "monotype", 79, 83, 100, 85, 78, [water_gun, tackle, water_gun, tackle]),
            Pokemon("Venusaur", "Grass", "Poison", 80, 82, 83, 100, 80, [vine_whip, tackle, vine_whip, tackle])
        ])

        self.done = False

        # Reset das estatísticas
        self.super_effective_count = 0
        self.not_very_effective_count = 0
        self.switch_count = 0
        self.total_damage_dealt = 0
        self.total_damage_received = 0
        self.pokemon_usage = Counter()

        return self._get_obs()

    def _get_obs(self):
        """Gera vetor de observação contínuo para todos os Pokémons e seus movimentos"""
        obs = []
        for p in [self.team1.active, *self.team1.pokemons, self.team2.active, *self.team2.pokemons]:
            obs.extend([p.hp, p.total_hp, p.attack, p.defense, p.special, p.speed])
            for m in p.moves:
                obs.extend([m.pp, m.total_pp, m.power, m.accuracy, m.priority])
                obs.extend(type_to_onehot(m.type))
            obs.extend(type_to_onehot(p.type1))
            obs.extend(type_to_onehot(p.type2) if p.type2 != "monotype" else [0]*len(used_types))
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        Executa uma ação do agente:
        - Move do Pokémon ativo ou troca
        - Atualiza estatísticas
        - Calcula recompensa baseada em dano e efetividade
        - Retorna nova observação, recompensa, done e info
        """
        if self.done:
            return self._get_obs(), 0, self.done, {}

        reward = 0
        active = self.team1.active

        info = {"super_effective": 0,
                "not_very_effective": 0,
                "switches": 0,
                "pokemon_usage": {},
                "damage_dealt": 0,
                "damage_received": 0}

        #AÇÃO DO AGENTE 
        if action < 4:
            dmg, eff = active.make_move(self.team2.active, action)
            self.total_damage_dealt += dmg if dmg else 0
            if eff == "SuperEffective":
                reward += 10
                self.super_effective_count += 1
            elif eff == "NotVeryEffective":
                reward -= 2
                self.not_very_effective_count += 1
            else:
                reward += int(dmg / 10)
        else:
            switch_idx = action - 4
            self.switch_count += 1
            if 0 <= switch_idx < len(self.team1.pokemons) and not self.team1.pokemons[switch_idx].fainted:
                self.team1.switch(switch_idx)
            else:
                reward -= 3

        # AÇÃO DO ADVERSÁRIO 
        alive_moves = [i for i, m in enumerate(self.team2.active.moves) if m.pp > 0]
        if alive_moves and random.random() < 0.7:
            move_idx = random.choice(alive_moves)
            dmg, _ = self.team2.active.make_move(self.team1.active, move_idx)
            self.total_damage_received += dmg if dmg else 0
        else:
            alive_idxs = [i for i, p in enumerate(self.team2.pokemons) if not p.fainted and p != self.team2.active]
            if alive_idxs:
                self.team2.switch(random.choice(alive_idxs))

        #  VERIFICAÇÕES DE FINAL
        if self.team1.lose():
            reward -= 50
            self.total_losses += 1
            self.done = True
        elif self.team2.lose():
            reward += 50
            self.total_wins += 1
            self.done = True

        # Contabilizando uso de Pokémon
        for p in self.team1.pokemons:
            if not p.fainted:
                self.pokemon_usage[p.name] += 1

        # Atualiza info
        info["super_effective"] = self.super_effective_count
        info["not_very_effective"] = self.not_very_effective_count
        info["switches"] = self.switch_count
        info["pokemon_usage"] = dict(self.pokemon_usage)
        info["damage_dealt"] = self.total_damage_dealt
        info["damage_received"] = self.total_damage_received

        return self._get_obs(), reward, self.done, info

# Função para discretizar observações
def discretize_obs(obs, bins=10):
    """
    Q-learning com tabela requer estados discretos.
    Esta função divide cada valor do vetor contínuo em 'bins' faixas discretas.
    """
    return tuple(int(x // (256/bins)) for x in obs)


# Treinamento Q-learning
"""
Explicação do algoritmo Q-learning:
- Q(s,a) representa a expectativa de recompensa tomando ação 'a' no estado 's'.
- A cada passo:
    1. Escolhemos ação usando política epsilon-greedy.
    2. Observamos recompensa e próximo estado.
- ε controla exploração vs exploração (epsilon-greedy).
"""

env = PokemonEnv()
episodes = 500
alpha = 0.1
gamma = 0.9
epsilon = 0.1
Q = {}

# Listas para métricas
wins_history = []
losses_history = []
super_effective_history = []
not_very_effective_history = []
switches_history = []
episode_reward_history = []
episode_pokemon_usage = []
damage_dealt_history = []
damage_received_history = []
pokemon_survivors_history = []

for ep in range(episodes):
    obs = env.reset()
    state = discretize_obs(obs)
    done = False
    total_reward = 0

    while not done:
        # Ações válidas
        active = env.team1.active
        valid_moves = [i for i in range(4) if active.moves[i].pp > 0]
        valid_switches = [i + 4 for i, p in enumerate(env.team1.pokemons) if p != active and not p.fainted]
        possible_actions = valid_moves + valid_switches
        if not possible_actions:
            break

        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(possible_actions)
        else:
            q_values = [Q.get((state, a), 0) for a in possible_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
            action = random.choice(best_actions)

        # Executa ação
        next_obs, reward, done, info = env.step(action)
        next_state = discretize_obs(next_obs)

        # Atualiza Q-table
        old_q = Q.get((state, action), 0)
        next_max = max([Q.get((next_state, a), 0) for a in possible_actions], default=0)
        Q[(state, action)] = old_q + alpha * (reward + gamma * next_max - old_q)

        state = next_state
        total_reward += reward

    # Estatísticas finais do episódio
    survivors = sum(not p.fainted for p in env.team1.pokemons)
    wins_history.append(env.total_wins)
    losses_history.append(env.total_losses)
    super_effective_history.append(info["super_effective"])
    not_very_effective_history.append(info["not_very_effective"])
    switches_history.append(info["switches"])
    episode_reward_history.append(total_reward)
    episode_pokemon_usage.append(info["pokemon_usage"])
    damage_dealt_history.append(info["damage_dealt"])
    damage_received_history.append(info["damage_received"])
    pokemon_survivors_history.append(survivors)

    if (ep+1) % 50 == 0:
        print(f"Episódio {ep+1}, Recompensa total: {total_reward}, Wins: {env.total_wins}, Losses: {env.total_losses}")


# Dados para os gráficos
total_episodes = len(wins_history)
win_rate = [100 * w / (w + l) if (w + l) > 0 else 0 for w, l in zip(wins_history, losses_history)]
avg_super_effective = np.cumsum(super_effective_history) / np.arange(1, total_episodes + 1)
avg_not_very_effective = np.cumsum(not_very_effective_history) / np.arange(1, total_episodes + 1)
avg_switches = np.cumsum(switches_history) / np.arange(1, total_episodes + 1)
avg_reward = np.cumsum(episode_reward_history) / np.arange(1, total_episodes + 1)
avg_damage_dealt = np.cumsum(damage_dealt_history) / np.arange(1, total_episodes + 1)
avg_damage_received = np.cumsum(damage_received_history) / np.arange(1, total_episodes + 1)
avg_survivors = np.cumsum(pokemon_survivors_history) / np.arange(1, total_episodes + 1)

# --- Gráficos ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, total_episodes+1), win_rate, label="Taxa de Vitórias (%)", color="green")
plt.title("Taxa de Vitórias ao Longo dos Episódios")
plt.xlabel("Episódio")
plt.ylabel("Taxa de Vitórias (%)")
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, total_episodes+1), avg_super_effective, label="Média Super Efetivo", color="blue")
plt.plot(range(1, total_episodes+1), avg_not_very_effective, label="Média Pouco Efetivo", color="orange")
plt.title("Efetividade Média por Episódio")
plt.xlabel("Episódio")
plt.ylabel("Média")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, total_episodes+1), avg_switches, label="Trocas Médias", color="purple")
plt.title("Trocas Médias ao Longo dos Episódios")
plt.xlabel("Episódio")
plt.ylabel("Média de Trocas")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, total_episodes+1), avg_reward, label="Recompensa Média", color="darkred")
plt.title("Recompensa Média ao Longo dos Episódios")
plt.xlabel("Episódio")
plt.ylabel("Recompensa Média")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(1, total_episodes+1), avg_damage_dealt, label="Dano Causado Médio", color="blue")
plt.plot(range(1, total_episodes+1), avg_damage_received, label="Dano Recebido Médio", color="red")
plt.title("Dano Médio por Episódio")
plt.xlabel("Episódio")
plt.ylabel("Dano Médio")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(1, total_episodes+1), avg_survivors, label="Pokémons Sobreviventes Médios", color="brown")
plt.title("Pokémons Sobreviventes Médios por Episódio")
plt.xlabel("Episódio")
plt.ylabel("Média de Sobreviventes")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
