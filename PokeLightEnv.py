import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
import csv

class PokeLightEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    # MÉTODOS DE AMBIENTE =================================================================================================
    def __init__(self, render_mode: str, max_hp: int, fps: int ):      
        self.max_hp = max_hp
        
        # ações disponíveis, atacar com um dos 6 pokémon
        self.action_space = spaces.Discrete(6)

        # espaço de observação
        self.vida_inicial = max_hp
        self.observation_space = spaces.Tuple((
            spaces.Discrete(6),  # tipo agente
            spaces.Discrete(6),  # tipo oponente
            spaces.Discrete(self.vida_inicial + 1),  # vida agente
            spaces.Discrete(self.vida_inicial + 1)   # vida oponente
        ))

        # pygame
        self.fps = fps
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.font_names = None
        self.window_size = (960, 640)

        # sprites
        self.sprites_agente = {
            0: pygame.image.load("graphics/blastoise_back.png"),
            1: pygame.image.load("graphics/charizard_back.png"),
            2: pygame.image.load("graphics/venosaur_back.png"),
            3: pygame.image.load("graphics/machamp_back.png"),
            4: pygame.image.load("graphics/gengar_back.png"),
            5: pygame.image.load("graphics/mewtwo_back.png"),
        }
        for k in self.sprites_agente:
            self.sprites_agente[k] = pygame.transform.scale(self.sprites_agente[k], (220, 220))

        self.sprites_oponente = {
            0: pygame.image.load("graphics/blastoise.png"),
            1: pygame.image.load("graphics/charizard.png"),
            2: pygame.image.load("graphics/venosaur.png"),
            3: pygame.image.load("graphics/machamp.png"),
            4: pygame.image.load("graphics/gengar.png"),
            5: pygame.image.load("graphics/mewtwo.png"),
        }
        for k in self.sprites_oponente:
            self.sprites_oponente[k] = pygame.transform.scale(self.sprites_oponente[k], (200, 200))

        self.background = pygame.image.load("graphics/Cenario.png")
        self.background = pygame.transform.scale(self.background, self.window_size)

        self.type_names = {0: "Blastoise", 1: "Charizard", 2: "Venosaur", 3: "Machamp", 4: "Gengar", 5: "Mewtwo"}
        self.nome_agente = "AGENTE"
        self.nome_oponente = "OPONENTE"
        self.battle_log = ""

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vida_agente = self.max_hp
        self.vida_oponente = self.max_hp

        # Sorteia os pokémon inciais, por parte do agente é meramente ilustrativo
        self.tipo_agente = self.action_space.sample()
        self.tipo_oponente = self.action_space.sample()

        self.battle_log = "A batalha começou!"
        return self._get_obs(), {}
    
    def _get_obs(self):
            return np.array([self.tipo_agente, self.tipo_oponente, self.vida_agente, self.vida_oponente], dtype=np.int32)
    
    # MÉTODOS DE JOGO ===================================================================================================== 
    def step(self, action):
        # só inicia pygame se for renderizar
        if self.render_mode == "human":
            self.init_pygame()

        # Ação do agente
        self.tipo_agente = int(action)
        if self.render_mode == "human":
            self.animacao_ataque("agente")
        dano_agente, msg = self.calcular_dano(self.tipo_agente, self.tipo_oponente)
        self.vida_oponente = max(self.vida_oponente - dano_agente, 0)

        self.battle_log = f"{self.type_names[self.tipo_agente]} do {self.nome_agente} atacou {self.type_names[self.tipo_oponente]} do {self.nome_oponente}\n{msg}"

        # Recompensa baseada na efetividade
        if dano_agente == 4:
            reward = 2
        elif dano_agente == 2:
            reward = -5
        else:
            reward = 0.0

        # Checa vitória do agente
        if self.vida_oponente <= 0:
            reward += 10
            return self._get_obs(), reward, True, False, {}

        # Oponente ataca
        dano_oponente, msg_op = self.calcular_dano(self.tipo_oponente, self.tipo_agente)
        if self.render_mode == "human":
            self.animacao_ataque("oponente")
        # Oponente baseado em RNG
        next_pokemon = int(self.np_random.integers(0, 6))
        if self.render_mode == "human" and next_pokemon != self.tipo_oponente:
            self.animacao_troca("oponente", next_pokemon)
        self.tipo_oponente = next_pokemon
        self.vida_agente = self.vida_agente - dano_oponente
        self.battle_log += f"\n{self.type_names[self.tipo_oponente]} do {self.nome_oponente} atacou {self.type_names[self.tipo_agente]} do {self.nome_agente}\n{msg_op}"

        # Checa derrota
        if self.vida_agente <= 0:
            reward -= 30
            return self._get_obs(), reward, True, False, {}

        return self._get_obs(), reward, False, False, {}
    
    def calcular_dano(self, atacante, defensor):
    # Matriz de efetividade
    # 4 = super efetivo, 3 = neutro, 2 = pouco efetivo
        efetividade = np.array([
            # 0   1   2   3   4   5
            [3,  4,  2,  3,  3,  3],  # Blastoise
            [2,  3,  4,  3,  3,  3],  # Charizard
            [4,  2,  3,  3,  3,  3],  # Venosaur
            [3,  3,  3,  3,  4,  2],  # Machamp
            [3,  3,  3,  2,  3,  4],  # Gengar
            [3,  3,  3,  4,  2,  3],  # Mewtwo
        ])

        dano = efetividade[atacante, defensor]

        if dano == 4:
            msg = "É super efetivo!"
        elif dano == 3:
            msg = "É neutro."
        elif dano == 2:
            msg = "Não é muito efetivo."

        return dano, msg

    # MÉTODOS DE GRÁFICOS =================================================================================================
    def init_pygame(self):
        if self.screen is None:
            pygame.init()
            self.font = pygame.font.Font("graphics/PokemonGB.otf", 28)
            self.font_names = pygame.font.Font("graphics/PokemonGB.otf", 22)
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("PokeLight")
            self.clock = pygame.time.Clock()

    def render_frame(self, agente_sprite=None, oponente_sprite=None, agente_pos=None, oponente_pos=None):
        
        self.screen.blit(self.background, (0, 0))
        
        # parece redundante, mas é usado nas animações
        if agente_sprite is None:
            agente_sprite = self.sprites_agente[self.tipo_agente]
        if oponente_sprite is None:
            oponente_sprite = self.sprites_oponente[self.tipo_oponente]
        if agente_pos is None:
            agente_pos = [120, 228]
        if oponente_pos is None:
            oponente_pos = [610, 90]

        self.screen.blit(agente_sprite, agente_pos)
        self.screen.blit(oponente_sprite, oponente_pos)

        # nomes e barras de vida
        self.render_texto(self.nome_agente, 565, 310)
        self.render_texto(self.nome_oponente, 75, 70)
        self.render_barra_vida(696, 367, self.vida_agente, self.max_hp, 192, 13)
        self.render_barra_vida(207, 131, self.vida_oponente, self.max_hp, 193, 13)
        self.render_texto(f"{self.vida_agente}/{self.max_hp}", 785, 380)
        self.render_texto(f"{self.vida_oponente}/{self.max_hp}", 280, 72)

        # mensagens de batalha
        y_offset = 475
        for i, line in enumerate(self.battle_log.split("\n")):
            shadow_text = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(shadow_text, (52, y_offset + i * 30 + 2))
            main_text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(main_text, (50, y_offset + i * 30))
        pygame.display.flip()
        self.clock.tick(self.fps)
        
    # Sequencia de funções de animações completamente gambiarras, mas funcionais, peço perdão, estamos sem tempo
    def animacao_ataque(self, alvo="agente", deslocamento=20, steps=5):
        if alvo == "agente":
            sprite = self.sprites_agente[self.tipo_agente]
            pos = [120, 228]
        else:
            sprite = self.sprites_oponente[self.tipo_oponente]
            pos = [610, 90]

        for i in range(steps):
            dx = (deslocamento / steps) * (i + 1)
            if alvo == "agente":
                self.render_frame(agente_sprite=sprite, agente_pos=[pos[0] + dx, pos[1]])
            else:
                self.render_frame(oponente_sprite=sprite, oponente_pos=[pos[0] - dx, pos[1]])
        for i in range(steps):
            dx = deslocamento - (deslocamento / steps) * (i + 1)
            if alvo == "agente":
                self.render_frame(agente_sprite=sprite, agente_pos=[pos[0] + dx, pos[1]])
            else:
                self.render_frame(oponente_sprite=sprite, oponente_pos=[pos[0] - dx, pos[1]])

    def animacao_troca(self, alvo="agente", novo_tipo=None, steps=10):
        if alvo == "agente":
            sprite_atual = self.sprites_agente[self.tipo_agente]
            pos_base = [120, 228]
        else:
            sprite_atual = self.sprites_oponente[self.tipo_oponente]
            pos_base = [610, 90]

        for i in range(steps):
            scale = 1 - (i / steps) * 0.5
            sprite_scaled = pygame.transform.scale(sprite_atual,
                                                   (int(sprite_atual.get_width() * scale), int(sprite_atual.get_height() * scale)))
            alpha = 255 if i % 2 == 0 else 100
            sprite_scaled.set_alpha(alpha)
            if alvo == "agente":
                self.render_frame(agente_sprite=sprite_scaled, agente_pos=pos_base)
            else:
                self.render_frame(oponente_sprite=sprite_scaled, oponente_pos=pos_base)

        if novo_tipo is not None:
            if alvo == "agente":
                self.tipo_agente = novo_tipo
                sprite_novo = self.sprites_agente[novo_tipo]
            else:
                sprite_novo = self.sprites_oponente[novo_tipo]
        else:
            sprite_novo = sprite_atual

        for i in range(steps):
            scale = 0.5 + (i / steps) * 0.5
            sprite_scaled = pygame.transform.scale(sprite_novo,
                                                   (int(sprite_novo.get_width() * scale), int(sprite_novo.get_height() * scale)))
            alpha = 100 if i % 2 == 0 else 255
            sprite_scaled.set_alpha(alpha)
            if alvo == "agente":
                self.render_frame(agente_sprite=sprite_scaled, agente_pos=pos_base)
            else:
                self.render_frame(oponente_sprite=sprite_scaled, oponente_pos=pos_base)
                
    # essa aqui ficou bonitinha, igualzinho aos jogos de Game Boy Advanced
    def render_barra_vida(self, x, y, hp, hp_total, width=100, height=15):
        percent = hp / hp_total
        if percent > 0.7:
            color = (0, 255, 0)
        elif percent > 0.3:
            color = (255, 255, 0)
        else:
            color = (255, 0, 0)
        pygame.draw.rect(self.screen, color, (x, y, int(width * percent), height))

    def render_texto(self, texto_par, x, y):
        texto = self.font.render(texto_par, True, (64, 64, 64))
        sombra = self.font.render(texto_par, True, (128, 128, 128))
        self.screen.blit(sombra, (x + 2, y + 2))
        self.screen.blit(texto, (x, y))

    def render(self):
        self.init_pygame()
        self.render_frame()

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

class ValueIterationAgent:
    def __init__(self, env, max_hp):
        self.env = env
        self.max_hp = max_hp
        # tamanhozinho do numero total de estados, kkkk
        self.nS = (max_hp + 1) ** 2 * 6 * 6
        self.nA = env.action_space.n
        self.V = np.zeros(self.nS)
        self.policy = np.zeros((self.nS, self.nA))

    def state_to_index(self, hp_a, hp_o, tipo_a, tipo_o):
        # transforma os dados do estado em um indice
        return ((hp_a * (self.max_hp + 1) + hp_o) * 6 + tipo_a) * 6 + tipo_o

    def index_to_state(self, idx):
        # transforma o numero do indice em dados do estado
        tipo_o = idx % 6
        idx //= 6
        tipo_a = idx % 6
        idx //= 6
        hp_o = idx % (self.max_hp + 1)
        idx //= (self.max_hp + 1)
        hp_a = idx
        return hp_a, hp_o, tipo_a, tipo_o

    def one_step_lookahead(self, state_idx, discount_factor=1.0):
        
        hp_a, hp_o, tipo_a, tipo_o = self.index_to_state(state_idx)
        
        # A vector of length env.nA containing the expected value of each action.
        # Igual o do notebook
        A = np.zeros(self.nA)

        # Itera cada ação possível
        for a in range(self.nA):
            
            # Simula os as escolhas
            dano_agente, _ = self.env.calcular_dano(a, tipo_o)
            hp_o_novo = max(hp_o - dano_agente, 0)

            if dano_agente == 4:
                reward = 2
            elif dano_agente == 2:
                reward = -5
            else:
                reward = 0
            if hp_o_novo == 0:
                A[a] = reward + 10
                continue

            dano_oponente, _ = self.env.calcular_dano(tipo_o, a)
            hp_a_novo = max(hp_a - dano_oponente, 0)

            if hp_a_novo == 0:
                A[a] = reward - 30
                continue
            
            # Incerteza sobre a ação do agente
            expected_V = 0.0
            for next_p in range(self.nA):
                idx_novo = self.state_to_index(hp_a_novo, hp_o_novo, a, next_p)
                expected_V += self.V[idx_novo]
            expected_V /= 6.0 # probabilidade uniforme de troca

            A[a] = reward + discount_factor * self.V[idx_novo]

        return A

    def run_policy(self, max_steps=100, delay=0.5):
        # Reinicia o ambiente e a recompensa acumulada
        obs, info = self.env.reset()
        accumulated_reward = 0
        
        for _ in range(max_steps):
            # pega o estado e vê o indice dele
            tipo_a, tipo_o, hp_a, hp_o = obs
            state_idx = self.state_to_index((hp_a), (hp_o), (tipo_a), (tipo_o))
            
            # realiza ação e guardo a recompensa
            action = (np.argmax(self.policy[state_idx]))
            obs, reward, done, trunc, info = self.env.step(action)
            accumulated_reward += reward
            
            # renderiza o ambiente
            self.env.render()
            time.sleep(delay)
            
            if done or trunc:
                break
            
        return accumulated_reward
   
    def run_value_iteration(self, theta=1e-6, discount_factor=1.0):
        episode = 0
        
        while True:
            delta = 0
            episode += 1

            # Atualiza todos os estados
            for s in range(self.nS):
                v = self.V[s]
                A = self.one_step_lookahead(s, discount_factor)
                best_action_value = np.max(A)
                self.V[s] = best_action_value
                delta = max(delta, abs(v - best_action_value))

            # Debug a cada 10 iterações
            if episode % 10 == 0:
                print(f"Iteração {episode}, delta = {delta}")

            # Critério de parada
            if delta < theta:
                break

        # Extração da política ótima
        for s in range(self.nS):
            A = self.one_step_lookahead(s, discount_factor)
            best_action = np.argmax(A)
            self.policy[s] = np.eye(self.nA)[best_action]

        return self.policy, self.V

    
    def export_policy_to_csv(self, filename="policy.csv"):
        # Exporta a politica para um CSV, estava curioso.
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Blastoise", "Charizard", "Venosaur", "Machamp", "Gengar", "Mewtwo"])
            for s in range(self.nS):
                writer.writerow(self.policy[s])
        print(f"Policy exportada para {filename}")

if __name__ == "__main__":
    max_hp = 50
    env = PokeLightEnv(render_mode="human", max_hp=max_hp, fps=30)
    
    vi_agent = ValueIterationAgent(env, max_hp=max_hp)
    policy, V = vi_agent.run_value_iteration()
    
    vi_agent.export_policy_to_csv("policy.csv")
    
    acc_reward = vi_agent.run_policy()
    print("Recompensa acumulada:", acc_reward)
    env.close()




