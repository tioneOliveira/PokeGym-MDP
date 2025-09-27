import random
import gymnasium as gym
from gym import spaces
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Tipos utilizados no experimento para reduzir dimensionalidade.
used_types = ["Normal", "Fire", "Water", "Grass", "Flying", "Poison"]

# Para cada tipo (defensor), listamos:
# - 'weaknesses': move types que são SUPER-EFETIVOS contra esse tipo (x2)
# - 'resistances': move types que são POUCO EFETIVOS contra esse tipo (x0.5)
# - 'immunities': move types que não afetam esse tipo (x0)
pokemon_types = {
    "Normal": {"weaknesses": [],              "resistances": [], "immunities": []},
    "Fire":   {"weaknesses": ["Water"],       "resistances": ["Grass"], "immunities": []},
    "Water":  {"weaknesses": ["Grass"],       "resistances": ["Fire"],  "immunities": []},
    "Grass":  {"weaknesses": ["Fire"],        "resistances": ["Water"], "immunities": []},
    "Flying": {"weaknesses": [] ,             "resistances": ["Grass"], "immunities": []},
    "Poison": {"weaknesses": ["Ground"],      "resistances": ["Grass"], "immunities": []}
}

def type_to_onehot(t, type_set=used_types):
    """Converte um tipo para vetor one-hot em relação a used_types."""
    vec = [0] * len(type_set)
    if t in type_set:
        vec[type_set.index(t)] = 1
    return vec


# Classes Move e Pokemon
class Move:
    """Representa um movimento (nome, tipo, stat, power, accuracy, pp, priority)."""
    def __init__(self, name="Tackle", mtype="Normal", stat="physical",
                 power=35, accuracy=95, pp=35, priority=0):
        self.name = name
        self.type = mtype
        self.stat = stat  # "physical" ou "special"
        self.power = power
        self.accuracy = accuracy
        self.total_pp = pp
        self.pp = pp
        self.priority = priority

    def reduce_pp(self):
        """Consume 1 PP se houver."""
        if self.pp > 0:
            self.pp -= 1

class Pokemon:
    """Representa um Pokémon com stats e 4 movimentos."""
    def __init__(self, name="Rattata", type1="Normal", type2="monotype",
                 hp=30, attack=56, defense=35, special=25, speed=72, moves=None):
        self.name = name
        self.type1 = type1
        self.type2 = type2  # "monotype" quando não tem segundo tipo
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
        Retorna multiplicador de tipo:
          0.0 se imunidade,
          2.0 se super efetivo,
          0.5 se pouco efetivo,
          1.0 caso normal.
        Observação: usa a tabela 'pokemon_types' (defensor-centric).
        """
        modifier = 1.0
        # itera sobre os tipos do Pokémon defensor
        for t in ([self.type1, self.type2] if self.type2 != "monotype" else [self.type1]):
            # se tipo não estiver no dicionário reduzimos a influência (compatibilidade reduzida)
            if t not in pokemon_types:
                continue
            info = pokemon_types[t]
            if move_type in info.get("immunities", []):
                return 0.0
            if move_type in info.get("weaknesses", []):
                modifier *= 2.0
            elif move_type in info.get("resistances", []):
                modifier *= 0.5
        return modifier

    def take_damage(self, dmg):
        """Aplica dano; se hp chega a zero marca fainted."""
        self.hp = max(0, self.hp - int(dmg))
        if self.hp == 0:
            self.fainted = True

    def make_move(self, target, move):
        """
        Executa a mecânica de golpe usando o objeto Move diretamente
        """
        if not isinstance(move, Move):
            return None, "InvalidMove"

        # Se sem PP
        if move.pp <= 0:
            return None, "NoPP"

        # Consumir PP
        move.reduce_pp()

        # Checar acurácia
        if random.randint(1, 100) > move.accuracy:
            return 0, "Miss"

        # Ataque físico vs especial
        atk_stat = self.attack if move.stat == "physical" else self.special
        def_stat = target.defense if move.stat == "physical" else target.special

        # Modificadores
        type_mod = target.calculate_type_modifier(move.type)
        stab = 1.5 if move.type in [self.type1, self.type2] else 1.0
        # Fórmula simplificada (nível fixo = 100)
        ratio = atk_stat / max(1, def_stat)
        base_damage = (((2 * 100 / 5) + 2) * move.power * ratio / 50 + 2)
        # adicionar variação aleatória 0.85 - 1.0
        base_damage *= (0.85 + random.random() * 0.15)
        damage = int(base_damage * type_mod * stab)

        # Aplica dano
        target.take_damage(damage)

        # Determina label do efeito
        if type_mod == 0.0:
            return 0, "NoEffect"
        if type_mod > 1.0:
            return damage, "SuperEffective"
        if 0 < type_mod < 1.0:
            return damage, "NotVeryEffective"
        return damage, "Normal"


class Team:
    """Time de pokémons com um ativo."""
    def __init__(self, pokemons, active_idx=0):
        self.pokemons = pokemons
        self.active = pokemons[active_idx]

    def switch(self, idx):
        """Troca Pokémon ativo (se válido e não desmaiado)."""
        if 0 <= idx < len(self.pokemons) and not self.pokemons[idx].fainted:
            self.active = self.pokemons[idx]

    def lose(self):
        """Retorna True se todos desmaiaram."""
        return all(p.fainted for p in self.pokemons)

# Ambiente Pokémon
class PokemonEnv(gym.Env):
    """
    Ambiente customizado:
    - action_space: Discrete(7)  => 0-3: ataques, 4-6: trocar para pokémon 0/1/2
    - observation: vetor com atributos dos 6 pokémon (ativo + time) e atributos dos movimentos.
    Regras chave:
      * Se pokémon ativo do agente estiver desmaiado no início do step, a única ação válida é switch.
      * Se não houver pokémon para trocar → derrota imediata.
      * Se agente escolher ação inválida → penalidade e retorno imediato do step (sem turno inimigo).
      * Depois da ação válida do agente (ataque ou troca válida), o inimigo age (a menos que a partida tenha terminado).
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(7)

        # construir tamanho do vetor de observação:
        # por pokémon: 6 stats + 4 moves * (5 attrs + len(used_types) one-hot) + 2 tipos one-hot
        per_pokemon_len = 6 + 4 * (5 + len(used_types)) + 2 * len(used_types)
        # temos 6 pokemons no total (3 do agent + 3 do inimigo)
        self.observation_space = spaces.Box(low=0, high=255, shape=(6 * per_pokemon_len,), dtype=np.float32)

        # métricas globais
        self.total_wins = 0
        self.total_losses = 0

        # métricas acumuladas
        self.total_damage_dealt = 0
        self.total_damage_received = 0

        # reset inicial
        self.reset()

    def reset(self):
        """Reinicia o ambiente: cria times fixos e zera métricas de episódio."""
        # definir movimentos base
        tackle = Move("Tackle", "Normal", "physical", 35, 95, 35)
        ember = Move("Ember", "Fire", "special", 40, 100, 25)
        water_gun = Move("Water Gun", "Water", "special", 40, 100, 25)
        vine_whip = Move("Vine Whip", "Grass", "physical", 45, 100, 25)
        quick_attack = Move("Quick Attack", "Normal", "physical", 40, 100, 20, 1)

        # criar times (Charizard, Blastoise, Venusaur)
        self.team1 = Team([
            Pokemon("Charizard", "Fire", "Flying", 312, 219, 207, 221, 251, [ember, quick_attack, ember, tackle]),
            Pokemon("Blastoise", "Water", "monotype", 314, 217, 251, 221, 207, [water_gun, quick_attack, water_gun, tackle]),
            Pokemon("Venusaur", "Grass", "Poison", 316, 215, 217, 251, 211, [vine_whip, quick_attack, vine_whip, tackle])
        ])
        self.team2 = Team([
            Pokemon("Charizard", "Fire", "Flying", 312, 219, 207, 221, 251, [ember, quick_attack, ember, tackle]),
            Pokemon("Blastoise", "Water", "monotype", 314, 217, 251, 221, 207, [water_gun, quick_attack, water_gun, tackle]),
            Pokemon("Venusaur", "Grass", "Poison", 316, 215, 217, 251, 211, [vine_whip, quick_attack, vine_whip, tackle])
        ])

        self.done = False

        # métricas de episódio
        self.episode_damage_dealt = 0
        self.episode_damage_received = 0
        self.episode_super_effective = 0
        self.episode_not_very_effective = 0
        self.episode_switches = 0
        self.episode_pokemon_usage = Counter()

        return self._get_obs()

    def _get_obs(self):
        """Constroi o vetor de observação (float32)."""
        obs = []
        for p in [self.team1.active, *self.team1.pokemons, self.team2.active, *self.team2.pokemons]:
            # stats básicos
            obs.extend([p.hp, p.total_hp, p.attack, p.defense, p.special, p.speed])
            # atributos dos movimentos
            for m in p.moves:
                obs.extend([m.pp, m.total_pp, m.power, m.accuracy, m.priority])
                # one-hot do tipo do movimento
                obs.extend(type_to_onehot(m.type))
            # tipos do pokémon (one-hot)
            obs.extend(type_to_onehot(p.type1))
            obs.extend(type_to_onehot(p.type2) if p.type2 != "monotype" else [0]*len(used_types))
        return np.array(obs, dtype=np.float32)

    def _valid_agent_switches(self):
        """Retorna índices (0..2) de pokémon que podem ser trocados (vivos e diferentes do ativo)."""
        return [i for i, p in enumerate(self.team1.pokemons) if not p.fainted and p != self.team1.active]

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, {}

        info = {}
        reward = 0

        # registrar uso do pokémon ativo (por turno)
        self.episode_pokemon_usage[self.team1.active.name] += 1

        # ---------- 1) FORCED SWITCH ----------
        if self.team1.active.fainted:
            valid_switches = self._valid_agent_switches()
            if not valid_switches:
                reward -= 100
                self.total_losses += 1
                self.done = True
                info["forced_switch"] = True
                return self._get_obs(), reward, self.done, info

            if action < 4:  # tentou atacar com pokémon morto
                reward -= 10
                info["invalid"] = True
                return self._get_obs(), reward, self.done, info

            switch_idx = action - 4
            if switch_idx not in valid_switches:
                reward -= 10
                info["invalid"] = True
                return self._get_obs(), reward, self.done, info

            self.team1.switch(switch_idx)
            self.episode_switches += 1
            info["forced_switch"] = True
            return self._get_obs(), reward, self.done, info

        # ---------- 2) AÇÃO DO AGENTE ----------
        agent_attack = None

        if action < 4:
            move = self.team1.active.moves[action]
            if move.pp <= 0:
                reward -= 3
                info["invalid"] = True
                return self._get_obs(), reward, self.done, info
            agent_attack = move
        else:
            switch_idx = action - 4
            if 0 <= switch_idx < len(self.team1.pokemons) and not self.team1.pokemons[switch_idx].fainted:
                if self.team1.pokemons[switch_idx] != self.team1.active:
                    self.team1.switch(switch_idx)
                    self.episode_switches += 1
            else:
                reward -= 3
                info["invalid"] = True
                return self._get_obs(), reward, self.done, info

        # ---------- 3) AÇÃO DO INIMIGO ----------
        enemy_attack = None
        if not self.done:
            alive_moves = [m for m in self.team2.active.moves if m.pp > 0]
            if alive_moves and random.random() < 0.7:
                enemy_attack = random.choice(alive_moves)
            else:
                alive_idxs = [i for i, p in enumerate(self.team2.pokemons) if not p.fainted and p != self.team2.active]
                if alive_idxs:
                    self.team2.switch(random.choice(alive_idxs))

        # ---------- 4) RESOLUÇÃO DE ATAQUES ----------
        actions_order = []

        if agent_attack and enemy_attack:
            if agent_attack.priority > enemy_attack.priority:
                reward += 2
                actions_order = [("agent", agent_attack), ("enemy", enemy_attack)]
            elif agent_attack.priority < enemy_attack.priority:
                actions_order = [("enemy", enemy_attack), ("agent", agent_attack)]
            else:
                if self.team1.active.speed >= self.team2.active.speed:
                    reward += 2
                    actions_order = [("agent", agent_attack), ("enemy", enemy_attack)]
                else:
                    actions_order = [("enemy", enemy_attack), ("agent", agent_attack)]
        elif agent_attack:
            actions_order = [("agent", agent_attack)]
        elif enemy_attack:
            actions_order = [("enemy", enemy_attack)]

        for actor, move in actions_order:
            if actor == "agent" and not self.team1.active.fainted:
                damage, effect = self.team1.active.make_move(self.team2.active, move)
                if damage:
                    reward += int(damage / 10)
                    self.episode_damage_dealt += damage
                if effect == "SuperEffective":
                    reward += 10
                    self.episode_super_effective += 1
                elif effect == "NotVeryEffective":
                    reward -= 2
                    self.episode_not_very_effective += 1

                if self.team2.active.fainted:
                    alive_enemy = [i for i, p in enumerate(self.team2.pokemons) if not p.fainted]
                    if alive_enemy:
                        self.team2.switch(random.choice(alive_enemy))
                    else:
                        reward += 100
                        self.total_wins += 1
                        self.done = True
                        self.total_damage_dealt += self.episode_damage_dealt
                        self.total_damage_received += self.episode_damage_received
                        info["ended_by"] = "agent_ko_all"
                        break

            elif actor == "enemy" and not self.team2.active.fainted and not self.done:
                damage, effect = self.team2.active.make_move(self.team1.active, move)
                if damage:
                    self.episode_damage_received += damage
                    reward -= int(damage / 10)

                if self.team1.active.fainted:
                    alive_agent = [i for i, p in enumerate(self.team1.pokemons) if not p.fainted]
                    if not alive_agent:
                        reward -= 50
                        self.total_losses += 1
                        self.done = True
                        self.total_damage_dealt += self.episode_damage_dealt
                        self.total_damage_received += self.episode_damage_received
                        info["ended_by"] = "agent_all_fainted"

        # ---------- 5) Atualiza métricas ----------
        info["super_effective"] = self.episode_super_effective
        info["not_very_effective"] = self.episode_not_very_effective
        info["switches"] = self.episode_switches
        info["pokemon_usage"] = dict(self.episode_pokemon_usage)
        info["damage_dealt"] = self.episode_damage_dealt
        info["damage_received"] = self.episode_damage_received
        info["forced_switch"] = False

        return self._get_obs(), reward, self.done, info



# Discretização para Q-table
def discretize_obs(obs, bins=6):
    """Discretiza o vetor de observação para usar em uma Q-table tabular.
    Aqui usamos divisão uniforme (quantização); retorna uma tupla (hashable)."""
    # obs pode ter valores variados; normalizamos entre 0 e 255 (observação já construída com esse range aproximado)
    # evitar divisão por zero
    obs = np.clip(obs, 0, 255)
    factor = 256 / bins
    discrete = (obs // factor).astype(int)
    return tuple(discrete.tolist())


# Treinamento Q-learning (com forced-switch alinhado)
env = PokemonEnv()
episodes = 10000
alpha = 0.1
gamma = 0.95
epsilon = 0.5

Q = {}  # dicionário (state_tuple, action) -> Q value

# listas para métricas por episódio
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
    state = discretize_obs(obs, bins=6)
    done = False
    total_reward = 0

    while not done:
        active = env.team1.active

        # --- determinar ações válidas ---
        if active.fainted:
            valid_actions = [4 + i for i, p in enumerate(env.team1.pokemons) if not p.fainted]
            if not valid_actions:
                break
        else:
            valid_moves = [i for i, m in enumerate(active.moves) if m.pp > 0]
            valid_switches = [4 + i for i, p in enumerate(env.team1.pokemons) if not p.fainted and p != active]
            valid_actions = valid_moves + valid_switches
            if not valid_actions:
                break

        # --- epsilon-greedy simplificado com NumPy ---
        if np.random.rand() < epsilon:
            # exploração: escolha aleatória entre ações válidas
            action = random.choice(valid_actions)
        else:
            # aproveitamento: escolha a ação com maior Q
            q_vals = np.array([Q.get((state, a), 0.0) for a in valid_actions])
            max_q = np.max(q_vals)
            # se houver empate, escolhe aleatoriamente entre as melhores
            best_actions = np.array(valid_actions)[q_vals == max_q]
            action = np.random.choice(best_actions)

        # --- executar ação ---
        next_obs, reward, done, info = env.step(action)
        next_state = discretize_obs(next_obs, bins=6)

        # --- Q-learning update ---
        # ações válidas no próximo estado
        next_active = env.team1.active
        if next_active.fainted:
            next_valid_actions = [4 + i for i, p in enumerate(env.team1.pokemons) if not p.fainted]
        else:
            nmoves = [i for i, m in enumerate(next_active.moves) if m.pp > 0]
            nswitches = [4 + i for i, p in enumerate(env.team1.pokemons) if not p.fainted and p != next_active]
            next_valid_actions = nmoves + nswitches

        # valor máximo futuro
        next_max = 0 if not next_valid_actions or done else np.max([Q.get((next_state, a), 0.0) for a in next_valid_actions])

        # atualizar Q
        old_q = Q.get((state, action), 0.0)
        Q[(state, action)] = old_q + alpha * (reward + gamma * next_max - old_q)

        # atualizar estado e recompensa
        state = next_state
        total_reward += reward

    # métricas do fim de episódio
    survivors = sum(1 for p in env.team1.pokemons if not p.fainted)
    wins_history.append(env.total_wins)
    losses_history.append(env.total_losses)
    super_effective_history.append(info.get("super_effective", 0))
    not_very_effective_history.append(info.get("not_very_effective", 0))
    switches_history.append(info.get("switches", 0))
    episode_reward_history.append(total_reward)
    episode_pokemon_usage.append(info.get("pokemon_usage", {}))
    damage_dealt_history.append(info.get("damage_dealt", 0))
    damage_received_history.append(info.get("damage_received", 0))
    pokemon_survivors_history.append(survivors)

    if (ep+1) % 50 == 0:
        print(f"Episódio {ep+1} | Recompensa episódio: {total_reward} | Wins: {env.total_wins} | "
            f"Losses: {env.total_losses} | Dano causado: {info['damage_dealt']} | "
            f"Dano recebido: {info['damage_received']} | Super efetivo: {info['super_effective']} | "
            f"Pouco efetivo: {info['not_very_effective']} | Sobreviventes: {sum(not p.fainted for p in env.team1.pokemons)}")



total_episodes = len(wins_history)
# taxa de vitórias cumulativa (em %)
win_rate = [100 * w / (w + l) if (w + l) > 0 else 0 for w, l in zip(wins_history, losses_history)]

avg_reward = np.cumsum(episode_reward_history) / np.arange(1, total_episodes + 1)
avg_damage_dealt = np.cumsum(damage_dealt_history) / np.arange(1, total_episodes + 1)
avg_damage_received = np.cumsum(damage_received_history) / np.arange(1, total_episodes + 1)
avg_survivors = np.cumsum(pokemon_survivors_history) / np.arange(1, total_episodes + 1)
avg_super = np.cumsum(super_effective_history) / np.arange(1, total_episodes + 1)
avg_notvery = np.cumsum(not_very_effective_history) / np.arange(1, total_episodes + 1)
avg_switches = np.cumsum(switches_history) / np.arange(1, total_episodes + 1)

plt.figure(figsize=(10,5))
plt.plot(range(1, total_episodes+1), win_rate, color="green")
plt.title("Taxa de Vitórias (cumulativa) %")
plt.xlabel("Episódio")
plt.ylabel("Vitórias (%)")
plt.ylim(0,100)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(1, total_episodes+1), avg_super, label="Super efetivo (média cumulativa)")
plt.plot(range(1, total_episodes+1), avg_notvery, label="Pouco efetivo (média cumulativa)")
plt.title("Efetividade média cumulativa")
plt.xlabel("Episódio")
plt.ylabel("Média cumulativa")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(1, total_episodes+1), avg_reward, color="darkred")
plt.title("Recompensa média cumulativa por episódio")
plt.xlabel("Episódio")
plt.ylabel("Recompensa média")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(1, total_episodes+1), avg_damage_dealt, label="Dano causado médio cumulativo")
plt.plot(range(1, total_episodes+1), avg_damage_received, label="Dano recebido médio cumulativo")
plt.title("Dano médio cumulativo")
plt.xlabel("Episódio")
plt.ylabel("Dano médio")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(1, total_episodes+1), avg_survivors, color="brown")
plt.title("Pokémons sobreviventes médios ao fim do episódio (cumulativo)")
plt.xlabel("Episódio")
plt.ylabel("Média de sobreviventes")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


plt.figure(figsize=(10,5))
plt.plot(range(1, total_episodes + 1), avg_switches, color="blue")
plt.title("Número médio de trocas por episódio (cumulativo)")
plt.xlabel("Episódio")
plt.ylabel("Trocas médias")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


all_pokemons = set()
for usage in episode_pokemon_usage:
    all_pokemons.update(usage.keys())

cumulative_usage = {p: np.zeros(total_episodes) for p in all_pokemons}

for ep_idx, usage in enumerate(episode_pokemon_usage):
    for p in all_pokemons:
        cumulative_usage[p][ep_idx] = usage.get(p, 0)

for p in cumulative_usage:
    cumulative_usage[p] = np.cumsum(cumulative_usage[p]) / (np.arange(1, total_episodes + 1))

plt.figure(figsize=(12,6))
for p, values in cumulative_usage.items():
    plt.plot(range(1, total_episodes+1), values, label=p)
plt.title("Uso médio cumulativo de cada Pokémon por episódio")
plt.xlabel("Episódio")
plt.ylabel("Uso médio")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()