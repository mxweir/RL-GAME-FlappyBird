# play.py
import pygame
from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import torch
import time
import logging
import os

def play():
    # Konfiguriere das Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Initialisiere die Umgebung mit Rendern
    env = FlappyBirdEnv(render_mode=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialisiere den Agent
    agent = DQNAgent(state_dim, action_dim)

    # Lade das trainierte Modell
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'dqn_flappy_bird.pth' if torch.cuda.is_available() else 'dqn_flappy_bird_cpu.pth'
    if os.path.exists(model_path):
        try:
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            logger.info(f"Modell erfolgreich geladen von {model_path}")
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            return
    else:
        logger.error(f"Trainiertes Modell nicht gefunden: {model_path}")
        return

    agent.policy_net.eval()  # Setze das Netzwerk in den Evaluierungsmodus

    # Setze Epsilon auf 0, um keine Exploration durchzuführen
    agent.epsilon_initial = 0.0
    agent.epsilon_end = 0.0
    agent.epsilon = 0.0

    num_episodes = 5  # Anzahl der Spiele, die gespielt werden sollen

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        score = 0
        logger.info(f"Starte Episode {episode}")
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            score = info['score']
            # Optional: Füge eine kurze Pause hinzu, um das Spiel visuell verfolgen zu können
            time.sleep(0.02)  # 20 ms

        logger.info(f"Episode {episode} beendet mit Score: {score}")
        print(f"Episode {episode}: Score = {score}")

    env.close()
    logger.info("Alle Episoden abgeschlossen.")

if __name__ == "__main__":
    play()
