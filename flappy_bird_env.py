# flappy_bird_env.py
import pygame
import gym
from gym import spaces
import numpy as np
import random

# Initialisiere pygame
pygame.init()

# Bildschirmgröße
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

# Farben
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Flugkraft und Gravitation
JUMP_STRENGTH = -10
GRAVITY = 0.5

# Pipe-Einstellungen
PIPE_WIDTH = 70
PIPE_GAP = 200
PIPE_SPEED = 3
PIPE_FREQUENCY = 1500  # Millisekunden

class FlappyBirdEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=False):
        super(FlappyBirdEnv, self).__init__()
        self.render_mode = render_mode
        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Flappy Bird DQN')
        self.clock = pygame.time.Clock()

        # Aktionen: 0 = nichts tun, 1 = springen
        self.action_space = spaces.Discrete(2)

        # Beobachtungsraum: [bird_y, bird_velocity, pipe_x, pipe_gap_y, distance_to_gap_center]
        self.observation_space = spaces.Box(
            low=np.array([0, -20, 0, 0, -SCREEN_HEIGHT / 2], dtype=np.float32),
            high=np.array([SCREEN_HEIGHT, 20, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_HEIGHT / 2], dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.bird_y = SCREEN_HEIGHT / 2
        self.bird_velocity = 0.0
        self.pipes = []
        self.score = 0
        self.done = False
        self.last_pipe_time = pygame.time.get_ticks()

        # Füge die erste Pipe hinzu
        pipe_height = random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
        self.pipes.append({'x': SCREEN_WIDTH, 'height': pipe_height, 'passed': False})

        return self._get_obs()

    def step(self, action):
        if action == 1:
            self.bird_velocity = JUMP_STRENGTH

        # Update Bird
        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        # Update Pipes
        current_time = pygame.time.get_ticks()
        if current_time - self.last_pipe_time > PIPE_FREQUENCY:
            pipe_height = random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
            self.pipes.append({'x': SCREEN_WIDTH, 'height': pipe_height, 'passed': False})
            self.last_pipe_time = current_time

        for pipe in self.pipes:
            pipe['x'] -= PIPE_SPEED

        # Entferne aus dem Bildschirm verschwundene Pipes
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] + PIPE_WIDTH > 0]

        # Überprüfe Kollisionen und Scoring
        reward = 0
        self.done = False

        # Kollision mit Boden oder Decke
        if self.bird_y <= 0 or self.bird_y >= SCREEN_HEIGHT:
            self.done = True
            reward = -10  # Bestrafung bei Tod

        # Kollision mit Pipes und Scoring
        for pipe in self.pipes:
            # Überprüfe Kollision mit der Pipe
            if pipe['x'] < 60 < pipe['x'] + PIPE_WIDTH:
                if not (pipe['height'] < self.bird_y < pipe['height'] + PIPE_GAP):
                    self.done = True
                    reward = -10  # Bestrafung bei Kollision

            # Überprüfe, ob Pipe erfolgreich passiert wurde
            if not pipe['passed'] and pipe['x'] + PIPE_WIDTH < 60:
                pipe['passed'] = True
                self.score += 1
                reward += 10  # Belohnung für das Passieren einer Pipe

        if not self.done:
            # Kleine Belohnung für jedes Überleben
            reward += 1

            # Zusätzliche Belohnung basierend auf der Nähe zur Mitte des Pipe-Gaps
            next_pipe = self.get_next_pipe()
            if next_pipe:
                pipe_gap_center = next_pipe['height'] + PIPE_GAP / 2
                distance_to_gap_center = (pipe_gap_center - self.bird_y)
                # Normalisierte Distanz, kleinere Distanz erhält höhere Belohnung
                distance_to_gap_center_norm = distance_to_gap_center / (SCREEN_HEIGHT / 2)
                reward += (SCREEN_HEIGHT / 2 - abs(distance_to_gap_center)) / (SCREEN_HEIGHT / 2)

        obs = self._get_obs()
        info = {'score': self.score}

        if self.render_mode:
            self.render()

        return obs, reward, self.done, info

    def render(self, mode='human'):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill(WHITE)

        # Zeichne den Vogel
        pygame.draw.circle(self.screen, BLACK, (60, int(self.bird_y)), 10)

        # Zeichne die Pipes
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(pipe['x'], 0, PIPE_WIDTH, pipe['height']))
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(pipe['x'], pipe['height'] + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT - pipe['height'] - PIPE_GAP))

        # Zeichne den Score
        font = pygame.font.SysFont(None, 36)
        score_surface = font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode:
            pygame.quit()

    def get_next_pipe(self):
        # Finde die nächste Pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + PIPE_WIDTH > 60:
                next_pipe = pipe
                break
        return next_pipe

    def _get_obs(self):
        # Finde die nächste Pipe
        next_pipe = self.get_next_pipe()

        if next_pipe is None:
            next_pipe = {'x': SCREEN_WIDTH, 'height': SCREEN_HEIGHT / 2 - PIPE_GAP / 2}

        # Normalisiere die Zustände
        bird_y_norm = self.bird_y / SCREEN_HEIGHT
        bird_velocity_norm = (self.bird_velocity + 10) / 20  # Annahme: velocity zwischen -10 und +10
        pipe_x_norm = next_pipe['x'] / SCREEN_WIDTH
        pipe_gap_y_norm = (next_pipe['height'] + PIPE_GAP / 2) / SCREEN_HEIGHT

        # Berechne die vertikale Distanz zur Mitte des Pipe-Gaps
        distance_to_gap_center = (next_pipe['height'] + PIPE_GAP / 2) - self.bird_y
        distance_to_gap_center_norm = distance_to_gap_center / (SCREEN_HEIGHT / 2)  # Normalisiert

        return np.array([
            bird_y_norm,
            bird_velocity_norm,
            pipe_x_norm,
            pipe_gap_y_norm,
            distance_to_gap_center_norm
        ], dtype=np.float32)
