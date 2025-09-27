import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time

# Constante de HP máximo
MAX_HP = 60

# Ambiente PokeLight
class PokeLightEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.MAX_HP = MAX_HP

        # Espaços
        self.observation_space = spaces.MultiDiscrete([self.MAX_HP+1, self.MAX_HP+1, 3, 3])
        self.action_space = spaces.Discrete(3)

        # Pygame
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.font_names = None
        self.window_size = (960, 640)

        # Sprites
        self.sprites_agente = {0: pygame.image.load("graphics/blastoise_back.png"),
                               1: pygame.image.load("graphics/charizard_back.png"),
                               2: pygame.image.load("graphics/venosaur_back.png")}
        for k in self.sprites_agente:
            self.sprites_agente[k] = pygame.transform.scale(self.sprites_agente[k], (220, 220))
        self.sprites_oponente = {0: pygame.image.load("graphics/blastoise.png"),
                                 1: pygame.image.load("graphics/charizard.png"),
                                 2: pygame.image.load("graphics/venosaur.png")}
        for k in self.sprites_oponente:
            self.sprites_oponente[k] = pygame.transform.scale(self.sprites_oponente[k], (200, 200))

        self.background = pygame.image.load("graphics/Cenario.png")
        self.background = pygame.transform.scale(self.background, self.window_size)

        self.type_names = {0: "Blastoise", 1: "Charizard", 2: "Venosaur"}
        self.nome_agente = "AGENTE"
        self.nome_oponente = "OPONENTE"
        self.battle_log = ""

        self.reset()

    # Funções de batalha
    def calcular_dano(self, atacante, defensor):
        if (atacante == 0 and defensor == 1) or \
           (atacante == 1 and defensor == 2) or \
           (atacante == 2 and defensor == 0):
            return 4, "É super efetivo!"
        elif atacante == defensor:
            return 3, "É neutro."
        else:
            return 2, "Não é muito efetivo."

    def step(self, action):
        self.init_pygame()

        # Agente ataca
        self.tipo_agente = action
        self.animacao_ataque("agente")
        
        dano_agente, msg = self.calcular_dano(self.tipo_agente, self.tipo_oponente)
        self.vida_oponente -= dano_agente
        
        self.battle_log = f"{self.type_names[self.tipo_agente]} do {self.nome_agente} atacou {self.type_names[self.tipo_oponente]} do {self.nome_oponente}\n{msg}"

        # Recompensa
        if "É super efetivo!" in msg:
            reward = 2
        elif "Não é muito efetivo." in msg:
            reward = -5
        else:
            reward = 0.0

        if self.vida_oponente <= 0:
            reward += 10
            return self._get_obs(), reward, True, False, {}

        # Oponente troca e ataca
        dano_oponente, msg = self.calcular_dano(self.tipo_oponente, self.tipo_agente)
        self.animacao_ataque("oponente")
        self.animacao_troca("oponente", (self.tipo_oponente + 1) % 3)
        self.vida_agente -= dano_oponente
        self.battle_log += f"\n{self.type_names[self.tipo_oponente]} do {self.nome_oponente} atacou {self.type_names[self.tipo_agente]} do {self.nome_agente}\n{msg}"

        if self.vida_agente <= 0:
            reward -= 30
            return self._get_obs(), reward, True, False, {}

        return self._get_obs(), reward, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vida_agente = self.MAX_HP
        self.vida_oponente = self.MAX_HP
        self.tipo_agente = 0
        self.tipo_oponente = 0
        self.battle_log = "A batalha começou!"
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.vida_agente, self.vida_oponente, self.tipo_agente, self.tipo_oponente], dtype=np.int32)

    # Renderização (não mexer)
    def init_pygame(self):
        if self.screen is None:
            pygame.init()
            self.font = pygame.font.Font("graphics/PokemonGB.otf", 28)
            self.font_names = pygame.font.Font("graphics/PokemonGB.otf", 22)
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("PokeLight")
            self.clock = pygame.time.Clock()

    # Renderização modular
    def render_frame(self, agente_sprite=None, oponente_sprite=None, agente_pos=None, oponente_pos=None):
        """Renderiza a tela com sprites opcionais"""
        self.screen.blit(self.background, (0, 0))

        if agente_sprite is None:
            agente_sprite = self.sprites_agente[self.tipo_agente]
        if oponente_sprite is None:
            oponente_sprite = self.sprites_oponente[self.tipo_oponente]
        if agente_pos is None:
            agente_pos = [120, 230]
        if oponente_pos is None:
            oponente_pos = [610, 90]

        self.screen.blit(agente_sprite, agente_pos)
        self.screen.blit(oponente_sprite, oponente_pos)

        # nomes e barras de vida
        self.render_texto(self.nome_agente, 565, 310)
        self.render_texto(self.nome_oponente, 75, 70)
        self.render_barra_vida(696, 367, self.vida_agente, MAX_HP, 192, 13)
        self.render_barra_vida(207, 131, self.vida_oponente, MAX_HP, 193, 13)
        self.render_texto(f"{self.vida_agente}/{MAX_HP}", 785, 380)
        self.render_texto(f"{self.vida_oponente}/{MAX_HP}", 280, 72)

        # mensagens de batalha com drop shadow
        y_offset = 480
        for i, line in enumerate(self.battle_log.split("\n")):
            shadow_text = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(shadow_text, (52, y_offset + i * 30 + 2))
            main_text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(main_text, (50, y_offset + i * 30))

        pygame.display.flip()
        self.clock.tick(30)

    # Animações
    def animacao_ataque(self, alvo="agente", deslocamento=20, steps=5):
        if alvo == "agente":
            sprite = self.sprites_agente[self.tipo_agente]
            pos = [120, 230]
        else:
            sprite = self.sprites_oponente[self.tipo_oponente]
            pos = [610, 90]

        for i in range(steps):
            dx = (deslocamento/steps) * (i+1)
            if alvo == "agente":
                self.render_frame(agente_sprite=sprite, agente_pos=[pos[0]+dx, pos[1]])
            else:
                self.render_frame(oponente_sprite=sprite, oponente_pos=[pos[0]-dx, pos[1]])
        for i in range(steps):
            dx = deslocamento - (deslocamento/steps)*(i+1)
            if alvo == "agente":
                self.render_frame(agente_sprite=sprite, agente_pos=[pos[0]+dx, pos[1]])
            else:
                self.render_frame(oponente_sprite=sprite, oponente_pos=[pos[0]-dx, pos[1]])

    def animacao_troca(self, alvo="agente", novo_tipo=None, steps=10):
        if alvo == "agente":
            sprite_atual = self.sprites_agente[self.tipo_agente]
            pos_base = [120, 230]
        else:
            sprite_atual = self.sprites_oponente[self.tipo_oponente]
            pos_base = [610, 90]

        # Encolher e brilho
        for i in range(steps):
            scale = 1 - (i/steps)*0.5
            sprite_scaled = pygame.transform.scale(sprite_atual,
                            (int(sprite_atual.get_width()*scale), int(sprite_atual.get_height()*scale)))
            alpha = 255 if i % 2 == 0 else 100
            sprite_scaled.set_alpha(alpha)
            if alvo == "agente":
                self.render_frame(agente_sprite=sprite_scaled, agente_pos=pos_base)
            else:
                self.render_frame(oponente_sprite=sprite_scaled, oponente_pos=pos_base)

        # Atualiza tipo
        if novo_tipo is not None:
            if alvo == "agente":
                self.tipo_agente = novo_tipo
                sprite_novo = self.sprites_agente[novo_tipo]
            else:
                self.tipo_oponente = novo_tipo
                sprite_novo = self.sprites_oponente[novo_tipo]
        else:
            sprite_novo = sprite_atual

        # Crescer e aparecer
        for i in range(steps):
            scale = 0.5 + (i/steps)*0.5
            sprite_scaled = pygame.transform.scale(sprite_novo,
                            (int(sprite_novo.get_width()*scale), int(sprite_novo.get_height()*scale)))
            alpha = 100 if i % 2 == 0 else 255
            sprite_scaled.set_alpha(alpha)
            if alvo == "agente":
                self.render_frame(agente_sprite=sprite_scaled, agente_pos=pos_base)
            else:
                self.render_frame(oponente_sprite=sprite_scaled, oponente_pos=pos_base)

    # Render modular de elementos
    def render_barra_vida(self, x, y, hp, hp_total, width=100, height=15):
        percent = hp / hp_total
        if percent > 0.7:
            color = (0, 255, 0)
        elif percent > 0.3:
            color = (255, 255, 0)
        else:
            color = (255, 0, 0)
        pygame.draw.rect(self.screen, color, (x, y, int(width*percent), height))

    def render_texto(self, texto_par, x, y):
        texto = self.font.render(texto_par, True, (64, 64, 64))
        sombra = self.font.render(texto_par, True, (128, 128, 128))
        self.screen.blit(sombra, (x+2, y+2))
        self.screen.blit(texto, (x, y))

    def render(self):
        self.init_pygame()
        self.render_frame()

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

# Value Iteration
def state_to_index(hp_a, hp_o, tipo_a, tipo_o, max_hp=MAX_HP):
    return ((hp_a * (max_hp+1) + hp_o) * 3 + tipo_a) * 3 + tipo_o

def index_to_state(idx, max_hp=MAX_HP):
    tipo_o = idx % 3
    idx //= 3
    tipo_a = idx % 3
    idx //= 3
    hp_o = idx % (max_hp+1)
    idx //= (max_hp+1)
    hp_a = idx
    return hp_a, hp_o, tipo_a, tipo_o

def one_step_lookahead(env, state_idx, V, discount_factor=1.0):
    hp_a, hp_o, tipo_a, tipo_o = index_to_state(state_idx)
    A = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):
        hp_o_novo, hp_a_novo = hp_o, hp_a
        dano_agente, _ = env.calcular_dano(a, tipo_o)
        hp_o_novo = max(hp_o - dano_agente, 0)

        # Recompensa
        if dano_agente == 4:
            reward = 2
        elif dano_agente == 2:
            reward = -5
        else:
            reward = 0

        done = False
        if hp_o_novo == 0:
            reward += 10
            done = True
        else:
            tipo_oponente_novo = (tipo_o + 1) % 3
            dano_oponente, _ = env.calcular_dano(tipo_oponente_novo, tipo_a)
            hp_a_novo = max(hp_a - dano_oponente, 0)
            if hp_a_novo == 0:
                reward -= 30
                done = True

        if done:
            A[a] = reward
        else:
            idx_novo = state_to_index(hp_a_novo, hp_o_novo, tipo_a, tipo_oponente_novo)
            A[a] = reward + discount_factor * V[idx_novo]

    return A

def value_iteration(env, theta=1e-6, discount_factor=1.0):
    max_hp = MAX_HP
    nS = (max_hp+1)**2 * 3 * 3
    nA = env.action_space.n

    V = np.zeros(nS)
    policy = np.zeros((nS, nA))

    delta = float('inf')
    while delta > theta:
        delta = 0
        for s in range(nS):
            v = V[s]
            A = one_step_lookahead(env, s, V, discount_factor)
            best_action_value = np.max(A)
            best_action = np.argmax(A)
            V[s] = best_action_value
            policy[s] = 0
            policy[s, best_action] = 1.0
            delta = max(delta, abs(v - V[s]))
    return policy, V

def run_policy(env, policy, max_steps=100):
    obs, info = env.reset()
    accumulated_reward = 0
    for _ in range(max_steps):
        hp_a, hp_o, tipo_a, tipo_o = obs
        state_idx = state_to_index(hp_a, hp_o, tipo_a, tipo_o)
        action = np.argmax(policy[state_idx])
        obs, reward, done, trunc, info = env.step(action)
        accumulated_reward += reward
        env.render()
        time.sleep(0.5)
        if done or trunc:
            break
    return accumulated_reward

# Execução
env = PokeLightEnv()
policy, V = value_iteration(env)
acc_reward = run_policy(env, policy)
print("Recompensa acumulada:", acc_reward)
env.close()
