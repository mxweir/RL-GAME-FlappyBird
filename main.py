# main.py
import pygame
from game import Bird, Pipe, check_collision
from graphics import Graphics
from agent import QLearningAgent
import random
import numpy as np

def discretize_state(bird, pipes):
    # Discretize the state for the Q-learning agent
    bird_y = min(int(bird.y // 50), 11)  # Discretize bird height (600 / 50 = 12 bins)
    pipe_x = min(int((pipes[0].x - bird.x) // 50), 7)  # Discretize horizontal distance to pipe (400 / 50 = 8 bins)
    pipe_height = min(int(pipes[0].height // 60), 9)  # Discretize pipe height (600 / 60 = 10 bins)
    return bird_y * 80 + pipe_x * 10 + pipe_height  # Create a unique state index

def main():
    graphics = Graphics()
    bird = Bird(60, 300)
    pipes = [Pipe(400, random.randint(100, 300), 150)]
    pipe_speed = 3
    score = 0
    highscore = 0

    state_space_size = 960  # Number of possible discrete states (12 * 8 * 10)
    action_space_size = 2  # Flap or not flap
    agent = QLearningAgent(state_space_size, action_space_size)

    running = True
    game_started = True
    game_over = False
    state = discretize_state(bird, pipes)
    round_counter = 0  # Counter to track rounds for periodic saving

    while running:
        if game_over:
            game_over = False
            game_started = True
            bird = Bird(60, 300)
            pipes = [Pipe(400, random.randint(100, 300), 150)]
            if score > highscore:
                highscore = score
            score = 0
            state = discretize_state(bird, pipes)
            round_counter += 1

            # Save model every 25 rounds
            if round_counter % 25 == 0:
                agent.save("q_table_checkpoint.npy")

        event = graphics.handle_events()
        if event == False:
            running = False

        if game_started and not game_over:
            action = agent.choose_action(state)
            if action == 1:
                bird.jump()

            bird.move()
            for pipe in pipes:
                pipe.move(pipe_speed)
                if pipe.is_off_screen():
                    pipes.remove(pipe)
                    pipes.append(Pipe(400, random.randint(100, 300), 150))
                    score += 1

            reward = 1  # Positive reward for each step
            if check_collision(bird, pipes):
                reward = -100  # Negative reward for collision
                game_over = True

            next_state = discretize_state(bird, pipes)
            agent.learn(state, action, reward, next_state, game_over)
            state = next_state

        graphics.clear_screen()
        graphics.draw_bird(bird)
        graphics.draw_pipes(pipes)
        graphics.draw_score(score, highscore)
        if game_over:
            graphics.draw_game_over()
        graphics.update_display()

    pygame.quit()

if __name__ == "__main__":
    main()
