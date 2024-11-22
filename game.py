# game.py
import random
import pygame

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.gravity = 0.3  # Further reduced gravity to make falling slower
        self.velocity = 0
        self.jump_strength = -6  # Further reduced jump strength for more control

    def jump(self):
        self.velocity = self.jump_strength

    def move(self):
        self.velocity += self.gravity
        self.y += self.velocity

    def get_position(self):
        return self.x, self.y

class Pipe:
    def __init__(self, x, height, gap_size):
        self.x = x
        self.height = height
        self.gap_size = gap_size + 50  # Increased gap size to make it easier to pass through
        self.width = 60  # Increased pipe width to make collision detection simpler

    def move(self, speed):
        self.x -= speed * 0.8  # Reduced pipe speed to make the game easier

    def is_off_screen(self):
        return self.x < -self.width

    def get_rects(self):
        return (
            pygame.Rect(self.x, 0, self.width, self.height),
            pygame.Rect(self.x, self.height + self.gap_size, self.width, 600 - self.height - self.gap_size),
        )

def check_collision(bird, pipes):
    bird_rect = pygame.Rect(bird.x, bird.y, 34, 24)  # Assuming a bird size
    for pipe in pipes:
        top_rect, bottom_rect = pipe.get_rects()
        if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
            return True
    if bird.y < 0 or bird.y > 600:  # Assuming screen height of 600
        return True
    return False

pygame.display.set_icon(pygame.image.load('utils/favicon.ico'))
