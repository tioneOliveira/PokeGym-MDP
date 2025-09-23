import random
import gym
from gym import spaces
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

pokemon_types = {
    "Normal": {"advantages": [], "disadvantages": ["Fighting"], "immunities": ["Ghost"]},
    "Fire": {"advantages": ["Grass", "Ice", "Bug"], "disadvantages": ["Water", "Rock", "Electric"], "immunities": []},
    "Water": {"advantages": ["Fire", "Ground", "Rock"], "disadvantages": ["Electric", "Grass"], "immunities": []},
    "Electric": {"advantages": ["Water", "Flying"], "disadvantages": ["Ground"], "immunities": []},
    "Grass": {"advantages": ["Water", "Ground", "Rock"], "disadvantages": ["Fire", "Poison", "Flying", "Bug"], "immunities": []},
    "Ice": {"advantages": ["Grass", "Ground", "Flying", "Dragon"], "disadvantages": ["Fire", "Fighting", "Rock"], "immunities": []},
    "Fighting": {"advantages": ["Normal", "Ice", "Rock"], "disadvantages": ["Flying", "Psychic"], "immunities": []},
    "Poison": {"advantages": ["Grass", "Bug"], "disadvantages": ["Ground", "Psychic"], "immunities": []},
    "Ground": {"advantages": ["Fire", "Electric", "Poison", "Rock"], "disadvantages": ["Water", "Grass", "Ice"], "immunities": ["Electric"]},
    "Flying": {"advantages": ["Grass", "Fighting", "Bug"], "disadvantages": ["Electric", "Rock"], "immunities": []},
    "Psychic": {"advantages": ["Fighting", "Poison"], "disadvantages": ["Bug"], "immunities": []},
    "Bug": {"advantages": ["Grass", "Psychic"], "disadvantages": ["Fire", "Flying", "Rock"], "immunities": []},
    "Rock": {"advantages": ["Fire", "Ice", "Flying", "Bug"], "disadvantages": ["Water", "Grass", "Fighting", "Ground"], "immunities": []},
    "Ghost": {"advantages": ["Psychic", "Ghost"], "disadvantages": ["Ghost"], "immunities": ["Normal", "Fighting"]},
    "Dragon": {"advantages": ["Dragon"], "disadvantages": ["Ice", "Dragon"], "immunities": []}
}

class Move:
    def __init__(self, name="Tackle", mtype="Normal", stat="physical", power=35, accuracy=95, pp=35, priority=0):
        self.name = name
        self.type = mtype
        self.stat = stat
        self.power = power
        self.accuracy = accuracy
        self.total_pp = pp
        self.pp = pp
        self.priority = priority

    def reduce_pp(self):
        if self.pp > 0:
            self.pp -= 1

class Pokemon:
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
        modifier = 1.0
        for t in [self.type1, self.type2] if self.type2 != "monotype" else [self.type1]:
            info = pokemon_types[t]
            if move_type in info["immunities"]:
                return 0.0
            if move_type in info["disadvantages"]:
                modifier *= 2
            elif move_type in info["advantages"]:
                modifier *= 0.5
        return modifier

    def take_damage(self, dmg):
        self.hp = max(0, self.hp - dmg)
        if self.hp == 0:
            self.fainted = True

    def make_move(self, target, move_idx):
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
    def __init__(self, pokemons, active_idx=0):
        self.pokemons = pokemons
        self.active = pokemons[active_idx]

    def switch(self, idx):
        if 0 <= idx < len(self.pokemons) and not self.pokemons[idx].fainted:
            self.active = self.pokemons[idx]

    def lose(self):
        return all(p.fainted for p in self.pokemons)


class PokemonEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(7)  # 0-3 moves, 4-6 switch
        # Observação: HP + PP + speed + tipo ativo (simplificado)
        self.observation_space = spaces.Box(low=0, high=255, shape=(18,), dtype=np.float32)
        self.reset()

        # Estatísticas
        self.total_wins = 0
        self.total_losses = 0
        self.total_switches = 0
        self.super_effective_hits = 0
        self.not_very_effective_hits = 0
        self.pokemon_usage = Counter()

    def reset(self):
        # Criar movimentos
        tackle = Move("Tackle", "Normal", "physical", 35, 95, 35)
        ember = Move("Ember", "Fire", "special", 40, 100, 25)
        water_gun = Move("Water Gun", "Water", "special", 40, 100, 25)
        vine_whip = Move("Vine Whip", "Grass", "physical", 45, 100, 25)

        # Criar pokémons
        self.team1 = Team([Pokemon("Charizard", "Fire", "Flying", 78, 84, 78, 85, 100, [ember, tackle, ember, tackle]),
                           Pokemon("Blastoise", "Water", "monotype", 79, 83, 100, 85, 78, [water_gun, tackle, water_gun, tackle]),
                           Pokemon("Venusaur", "Grass", "Poison", 80, 82, 83, 100, 80, [vine_whip, tackle, vine_whip, tackle])])
        self.team2 = Team([Pokemon("Charizard", "Fire", "Flying", 78, 84, 78, 85, 100, [ember, tackle, ember, tackle]),
                           Pokemon("Blastoise", "Water", "monotype", 79, 83, 100, 85, 78, [water_gun, tackle, water_gun, tackle]),
                           Pokemon("Venusaur", "Grass", "Poison", 80, 82, 83, 100, 80, [vine_whip, tackle, vine_whip, tackle])])

        self.done = False
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for p in [self.team1.active, *self.team1.pokemons, self.team2.active, *self.team2.pokemons]:
            obs.extend([p.hp, p.total_hp, p.speed])
            for m in p.moves:
                obs.append(m.pp)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, {}

        reward = 0

        self.pokemon_usage[self.team1.active.name] += 1


        if action < 4:
            
            move_idx = action
            dmg, eff = self.team1.active.make_move(self.team2.active, move_idx)
            if eff == "SuperEffective":
                reward += 5
                self.super_effective_hits += 1
            elif eff == "NotVeryEffective":
                reward -= 5
                self.not_very_effective_hits += 1
            elif eff == "NoPP":
                reward -= 1
            if self.team2.active.fainted:
                alive = [i for i, p in enumerate(self.team2.pokemons) if not p.fainted]
                if alive:
                    self.team2.switch(random.choice(alive))
        else:
            switch_idx = action - 4
            self.team1.switch(switch_idx)
            self.total_switches += 1

        alive_moves = [i for i, m in enumerate(self.team2.active.moves) if m.pp > 0]
        if alive_moves and random.random() < 0.7:
            move_idx = random.choice(alive_moves)
            dmg, eff = self.team2.active.make_move(self.team1.active, move_idx)
            if eff == "SuperEffective":
                reward -= 5
                self.super_effective_hits += 0
            elif eff == "NotVeryEffective":
                reward += 5
            if self.team1.active.fainted:
                alive = [i for i, p in enumerate(self.team1.pokemons) if not p.fainted]
                if alive:
                    self.team1.switch(random.choice(alive))

        if self.team1.lose():
            reward -= 10
            self.total_losses += 1
            self.done = True
        elif self.team2.lose():
            reward += 10
            self.total_wins += 1
            self.done = True

        return self._get_obs(), reward, self.done, {}

env = PokemonEnv()
episodes = 100000

rewards_history = []
wins_history = []
losses_history = []
switches_history = []
super_effective_history = []
not_very_effective_history = []
pokemon_usage_history = []

for ep in range(episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        valid_moves = [i for i in range(4) if env.team1.active.moves[i].pp > 0]
        valid_switches = [i + 4 for i, p in enumerate(env.team1.pokemons) if p != env.team1.active and not p.fainted]
        possible_actions = valid_moves + valid_switches
        action = random.choice(possible_actions)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    rewards_history.append(total_reward)
    wins_history.append(env.total_wins)
    losses_history.append(env.total_losses)
    switches_history.append(env.total_switches)
    super_effective_history.append(env.super_effective_hits)
    not_very_effective_history.append(env.not_very_effective_hits)
    pokemon_usage_history.append(env.pokemon_usage.copy())



plt.figure(figsize=(12,6))
plt.plot(rewards_history, label="Recompensa total por batalha")
plt.xlabel("Episódio")
plt.ylabel("Recompensa")
plt.title("Recompensa ao longo das batalhas")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(wins_history, label="Vitórias acumuladas")
plt.plot(losses_history, label="Derrotas acumuladas")
plt.xlabel("Episódio")
plt.ylabel("Quantidade")
plt.title("Vitórias e Derrotas")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(switches_history, label="Trocas acumuladas")
plt.xlabel("Episódio")
plt.ylabel("Trocas")
plt.title("Número de trocas")
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(super_effective_history, label="Super efetivos")
plt.plot(not_very_effective_history, label="Pouco efetivos")
plt.xlabel("Episódio")
plt.ylabel("Quantidade")
plt.title("Golpes super efetivos e pouco efetivos")
plt.legend()
plt.show()

aggregate_usage = Counter()
for usage in pokemon_usage_history:
    aggregate_usage.update(usage)
plt.figure(figsize=(8,6))
plt.bar(aggregate_usage.keys(), aggregate_usage.values())
plt.xlabel("Pokémon")
plt.ylabel("Uso acumulado em turnos")
plt.title("Uso de Pokémon ao longo de todas as batalhas")
plt.show()
