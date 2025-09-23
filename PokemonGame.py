import random

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
        if dmg <= 0:
            return
        self.hp -= dmg
        if self.hp <= 0:
            self.hp = 0
            self.fainted = True

    def make_move(self, target, move_idx):
        move = self.moves[move_idx]
        if move.pp <= 0:
            print(f"{self.name} tried {move.name}, but no PP left!")
            return False

        print(f"{self.name} uses {move.name}!")
        move.reduce_pp()

        if random.randint(1, 100) > move.accuracy:
            print("It missed!")
            return True

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
            print(">>> Super effective! <<<")
        elif 0 < type_mod < 1:
            print(">>> Not very effective... <<<")
        elif type_mod == 0:
            print(">>> No effect! <<<")

        return True

    def stats_to_string(self):
        t = f"{self.name} ({self.type1}"
        if self.type2 != "monotype":
            t += f"/{self.type2}"
        t += f") HP: {self.hp}/{self.total_hp}"
        if self.fainted:
            t += " !!! FAINTED !!!"
        return t

    def moves_to_string(self):
        s = "Moves:\n"
        for i, m in enumerate(self.moves):
            s += f"[{i}] {m.name} ({m.type}, {m.stat}) POW:{m.power} ACC:{m.accuracy} PP:{m.pp}/{m.total_pp}\n"
        return s


class Team:
    def __init__(self, pokemons, active_idx=0):
        self.pokemons = pokemons
        self.active = pokemons[active_idx]

    def switch(self, idx):
        if 0 <= idx < len(self.pokemons) and not self.pokemons[idx].fainted:
            self.active = self.pokemons[idx]

    def lose(self):
        return all(p.fainted for p in self.pokemons)

    def __str__(self):
        return "\n".join(f"[{i}] {p.stats_to_string()}" for i, p in enumerate(self.pokemons))


class Battle:
    def __init__(self, team1, team2):
        self.team1 = team1
        self.team2 = team2

    def start(self):
        while not self.team1.lose() and not self.team2.lose():
            print("\n--- Current Status ---")
            print("Team 1 active:", self.team1.active.stats_to_string())
            print("Team 2 active:", self.team2.active.stats_to_string())

            choice = int(input("\nTeam 1: 0-Attack, 1-Switch: "))
            if choice == 0:
                print(self.team1.active.moves_to_string())
                move_idx = int(input("Choose move: "))
                t1_action = ("attack", move_idx)
            else:
                print(self.team1)
                switch_idx = int(input("Choose Pokémon to switch: "))
                t1_action = ("switch", switch_idx)

            if random.random() < 0.7:  
                move_idx = random.randint(0, len(self.team2.active.moves)-1)
                t2_action = ("attack", move_idx)
            else:
                alive_idxs = [i for i, p in enumerate(self.team2.pokemons) if not p.fainted and p != self.team2.active]
                if alive_idxs:
                    t2_action = ("switch", random.choice(alive_idxs))
                else:
                    move_idx = random.randint(0, len(self.team2.active.moves)-1)
                    t2_action = ("attack", move_idx)

            if t1_action[0] == "switch":
                self.team1.switch(t1_action[1])
                print("Team 1 switched Pokémon!")
            if t2_action[0] == "switch":
                self.team2.switch(t2_action[1])
                print("Team 2 switched Pokémon!")

            if t1_action[0] == "attack" and t2_action[0] == "attack":
                move1 = self.team1.active.moves[t1_action[1]]
                move2 = self.team2.active.moves[t2_action[1]]

                if (move1.priority > move2.priority or
                   (move1.priority == move2.priority and
                    (self.team1.active.speed > self.team2.active.speed or
                     (self.team1.active.speed == self.team2.active.speed and random.randint(0, 1) == 0)))):
                    self.team1.active.make_move(self.team2.active, t1_action[1])
                    if not self.team2.active.fainted:
                        self.team2.active.make_move(self.team1.active, t2_action[1])
                else:
                    self.team2.active.make_move(self.team1.active, t2_action[1])
                    if not self.team1.active.fainted:
                        self.team1.active.make_move(self.team2.active, t1_action[1])

            elif t1_action[0] == "attack":
                self.team1.active.make_move(self.team2.active, t1_action[1])
            elif t2_action[0] == "attack":
                self.team2.active.make_move(self.team1.active, t2_action[1])

            if self.team1.active.fainted and not self.team1.lose():
                print("Team 1 Pokémon fainted! Choose a replacement:")
                print(self.team1)
                idx = int(input("Choose: "))
                self.team1.switch(idx)
            if self.team2.active.fainted and not self.team2.lose():
                alive_idxs = [i for i, p in enumerate(self.team2.pokemons) if not p.fainted]
                self.team2.switch(random.choice(alive_idxs))
                print("Team 2 Pokémon fainted! AI switched Pokémon.")

        if self.team1.lose():
            print("\nTeam 2 wins!")
        else:
            print("\nTeam 1 wins!")

if __name__ == "__main__":
    random.seed()


    tackle = Move("Tackle", "Normal", "physical", 35, 95, 35)
    ember = Move("Ember", "Fire", "special", 40, 100, 25)
    water_gun = Move("Water Gun", "Water", "special", 40, 100, 25)
    vine_whip = Move("Vine Whip", "Grass", "physical", 45, 100, 25)


    charizard = Pokemon("Charizard", "Fire", "Flying", 78, 84, 78, 85, 100, [ember, tackle, ember, tackle])
    blastoise = Pokemon("Blastoise", "Water", "monotype", 79, 83, 100, 85, 78, [water_gun, tackle, water_gun, tackle])
    venusaur = Pokemon("Venusaur", "Grass", "Poison", 80, 82, 83, 100, 80, [vine_whip, tackle, vine_whip, tackle])

 
    team1 = Team([charizard, blastoise, venusaur], 0)
    team2 = Team([Pokemon("Charizard", "Fire", "Flying", 78, 84, 78, 85, 100, [ember, tackle, ember, tackle]),
                  Pokemon("Blastoise", "Water", "monotype", 79, 83, 100, 85, 78, [water_gun, tackle, water_gun, tackle]),
                  Pokemon("Venusaur", "Grass", "Poison", 80, 82, 83, 100, 80, [vine_whip, tackle, vine_whip, tackle])], 0)

    print("Escolha seu Pokémon inicial:")
    print(team1)
    idx = int(input("Choose: "))
    team1.switch(idx)

    battle = Battle(team1, team2)
    battle.start()
