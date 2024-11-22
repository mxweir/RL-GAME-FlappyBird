# graphics.py
import pygame
from game import Bird, Pipe

class Graphics:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 600))
        pygame.display.set_caption("Learn FlappyBird | by mxweir")
        self.clock = pygame.time.Clock()
        self.background_color = (135, 206, 235)
        self.bird_color = (255, 255, 0)
        self.pipe_color = (0, 255, 0)
        self.font = pygame.font.Font(None, 36)

    def draw_bird(self, bird):
        x, y = bird.get_position()
        pygame.draw.ellipse(self.screen, self.bird_color, (x, y, 34, 24))

    def draw_pipes(self, pipes):
        for pipe in pipes:
            top_rect, bottom_rect = pipe.get_rects()
            pygame.draw.rect(self.screen, self.pipe_color, top_rect)
            pygame.draw.rect(self.screen, self.pipe_color, bottom_rect)

    def draw_game_over(self):
        font = pygame.font.Font(None, 74)
        text = font.render("Game Over", True, (255, 0, 0))
        self.screen.blit(text, (100, 250))
        font = pygame.font.Font(None, 36)
        text = font.render("Press 'R' to Restart", True, (255, 255, 255))
        self.screen.blit(text, (85, 320))

    def draw_score(self, score, highscore):
        score_text = self.font.render(f"Score: {score}", True, (0, 0, 0))
        highscore_text = self.font.render(f"Highscore: {highscore}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(highscore_text, (250, 10))

    def update_display(self):
        pygame.display.flip()
        self.clock.tick(60)

    def clear_screen(self):
        self.screen.fill(self.background_color)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return True
                if event.key == pygame.K_r:
                    return "restart"
        return None