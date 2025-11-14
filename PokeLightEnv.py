import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
class PokeLightEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    # MÉTODOS DE AMBIENTE =================================================================================================
    def __init__(self, render_mode: str, max_hp: int, fps: int ):      
        self.max_hp = max_hp
        
        # ações disponíveis, atacar com um dos 6 pokémon
        self.action_space = spaces.Discrete(7)

        # espaço de observação
        self.vida_inicial = max_hp
        obs_low  = np.zeros(14, dtype=np.float32)
        obs_high = np.ones(14, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(14,),
            dtype=np.float32
        )

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
        self.vida_agente = [self.max_hp, self.max_hp, self.max_hp, self.max_hp, self.max_hp, self.max_hp]
        self.vida_oponente = [self.max_hp, self.max_hp, self.max_hp, self.max_hp, self.max_hp, self.max_hp]

        # Sorteia os pokémon inciais, por parte do agente é meramente ilustrativo
        self.tipo_agente = int(self.np_random.integers(0, 6))
        self.tipo_oponente = int(self.np_random.integers(0, 6))

        self.battle_log = "A batalha começou!"
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = np.array([
            self.tipo_agente / 5,
            self.tipo_oponente / 5,
            * (np.array(self.vida_agente) / self.max_hp),
            * (np.array(self.vida_oponente) / self.max_hp)
        ], dtype=np.float32)

        return obs
    
    def step(self, action):
        reward = 0
        # só inicia pygame se for renderizar
        if self.render_mode == "human":
            self.init_pygame()


        # garante flags padrão
        terminated = False
        truncated = False


        # AÇÃO DO AGENTE (sempre age primeiro)
        if action == 0:
        # ataque
            if self.vida_agente[self.tipo_agente] == 0:
                reward = -3
                return self._get_obs(), reward, terminated, truncated, {}
            if self.render_mode == "human":
                self.animacao_ataque("agente")


            dano_agente, msg = self.calcular_dano(self.tipo_agente, self.tipo_oponente)
            self.vida_oponente[self.tipo_oponente] = max(self.vida_oponente[self.tipo_oponente] - dano_agente, 0)


            self.battle_log = (
                f"{self.type_names[self.tipo_agente]} do {self.nome_agente} atacou "
                f"{self.type_names[self.tipo_oponente]} do {self.nome_oponente} "
                f"{msg}"
            )


            # Recompensa baseada na efetividade (mantive sua escala)
            if dano_agente == 4:
                reward += 2
            elif dano_agente == 2:
                reward -= 5


            # Checa vitória imediata do agente
            if np.all(np.array(self.vida_oponente) == 0):
                reward += 100
                return self._get_obs(), reward, True, False, {}


        else:
            # troca do agente
            target = action - 1

            # validações: pokémon vivo e não trocar para o mesmo
            if target < 0 or target >= 6 or self.vida_agente[target] == 0 or target == self.tipo_agente:
                reward = -3
                return self._get_obs(), reward, terminated, truncated, {}


            self.tipo_agente = target
            self.battle_log = f"Agente trocou para {self.type_names[self.tipo_agente]}"

        # prepara lista de ações válidas para oponente
        valid_op_actions = []
        # ataque sempre é uma opção se o pokémon ativo do oponente estiver vivo
        if self.vida_oponente[self.tipo_oponente] > 0:
            valid_op_actions.append(0)
        # swaps válidos
        for idx in range(6):
            if self.vida_oponente[idx] > 0 and idx != self.tipo_oponente:
                valid_op_actions.append(idx + 1) # codifica swap como 1..6


        # se não houver ações válidas, o oponente está derrotado
        if len(valid_op_actions) == 0:
            reward += 100
            return self._get_obs(), reward, True, False, {}


        # escolhe ação do oponente
        op_action = int(self.np_random.choice(valid_op_actions))
        if op_action == 0:
        # ataque do oponente
            if self.render_mode == "human":
                self.animacao_ataque("oponente")


            dano_oponente, msg = self.calcular_dano(self.tipo_oponente, self.tipo_agente)
            self.vida_agente[self.tipo_agente] = max(self.vida_agente[self.tipo_agente] - dano_oponente, 0)


            self.battle_log = (
            f"{self.type_names[self.tipo_oponente]} do {self.nome_oponente} atacou "
            f"{self.type_names[self.tipo_agente]} do {self.nome_agente} "
            f"{msg}"
            )


            # checa derrota do agente
            if np.all(np.array(self.vida_agente) == 0):
                reward -= 100
                return self._get_obs(), reward, True, False, {}


        else:
            # swap do oponente
            new_op_idx = op_action - 1
            # segurança: garantir que seja válido
            if self.vida_oponente[new_op_idx] > 0 and new_op_idx != self.tipo_oponente:
                self.tipo_oponente = new_op_idx
                self.battle_log = f"Oponente trocou para {self.type_names[self.tipo_oponente]}"
            else:
                # fallback: se algo inesperado acontecer, oponente atacará se possível
                if self.vida_oponente[self.tipo_oponente] > 0:
                    dano_oponente, msg = self.calcular_dano(self.tipo_oponente, self.tipo_agente)
                    self.vida_agente[self.tipo_agente] = max(self.vida_agente[self.tipo_agente] - dano_oponente, 0)
                    self.battle_log = (
                            f"{self.type_names[self.tipo_oponente]} do {self.nome_oponente} atacou "
                            f"{self.type_names[self.tipo_agente]} do {self.nome_agente} "
                            f"{msg}"
                        )
                    if np.all(np.array(self.vida_agente) == 0):
                        reward -= 100
                        return self._get_obs(), reward, True, False, {}


        print("Vida Agente: " + str(self.vida_agente))
        print("Ativo Agente: " + str(self.tipo_agente))
        print("Acao Agente: " + str(action))
        print("------------------------")
        print("Vida Oponente: " + str(self.vida_oponente))
        print("Ativo Oponente: " + str(self.tipo_oponente))
        print("Acao Oponente: " + str(op_action))
        print("||||||||||||||||||||||||")
        print(self._get_obs())
        print("========================")
        
        
        return self._get_obs(), reward, terminated, truncated, {}
    
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
        self.render_barra_vida(696, 367, self.vida_agente[self.tipo_agente], self.max_hp, 192, 13)
        self.render_barra_vida(207, 131, self.vida_oponente[self.tipo_oponente], self.max_hp, 193, 13)
        self.render_texto(f"{self.vida_agente[self.tipo_agente]}/{self.max_hp}", 785, 380)
        self.render_texto(f"{self.vida_oponente[self.tipo_oponente]}/{self.max_hp}", 280, 72)

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


if __name__ == "__main__":
    max_hp = 10
    env = PokeLightEnv(render_mode="human", max_hp=max_hp, fps=60)
    
    done = False
    obs, _ = env.reset()
    total_reward = 0

    while not done:
        action = env.action_space.sample() 
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    print("Recompensa acumulada:", total_reward)
    env.close()




