# train_dqn.py
import os
import pickle
import matplotlib.pyplot as plt
from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import torch
import time
import logging
from torch.utils.tensorboard import SummaryWriter  # Importiere TensorBoard

def train():
    # Konfiguriere das Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Initialisiere die Umgebung ohne Rendern
    env = FlappyBirdEnv(render_mode=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    # Pfade für gespeichertes Modell und Replay-Puffer
    model_path = 'dqn_flappy_bird.pth' if torch.cuda.is_available() else 'dqn_flappy_bird_cpu.pth'
    memory_path = 'replay_buffer.pkl'

    # Überprüfe, ob ein gespeichertes Modell existiert und lade es
    if os.path.exists(model_path):
        try:
            agent.load_model(model_path)
            logger.info(f"Modell erfolgreich geladen von {model_path}")
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
    else:
        logger.info("Kein gespeichertes Modell gefunden. Starte mit einem neuen Modell.")

    # Überprüfe, ob ein gespeicherter Replay-Puffer existiert und lade ihn
    if os.path.exists(memory_path):
        try:
            with open(memory_path, 'rb') as f:
                agent.memory = pickle.load(f)
            logger.info(f"Replay-Puffer erfolgreich geladen von {memory_path}")
        except Exception as e:
            logger.error(f"Fehler beim Laden des Replay-Puffers: {e}")
    else:
        logger.info("Kein gespeicherter Replay-Puffer gefunden. Starte mit einem leeren Puffer.")

    num_episodes = 1000
    scores = []
    avg_scores = []
    losses = []

    # Initialisiere TensorBoard
    writer = SummaryWriter('runs/flappy_bird_dqn')

    start_time = time.time()
    logger.info("Starte das Training des DQN-Agenten...")

    try:
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
                    writer.add_scalar('Loss/train', loss, agent.steps_done)  # Logge den Verlust
                state = next_state
                total_reward += reward

                # Aktualisiere das Zielnetzwerk regelmäßig
                if agent.steps_done % agent.target_update_steps == 0:
                    agent.update_target_network()

            scores.append(info['score'])

            # Durchschnittlichen Score berechnen für TensorBoard alle 10 Episoden
            if episode % 10 == 0:
                avg_score = sum(scores[-10:]) / 10
                writer.add_scalar('Score/avg', avg_score, episode)
                logger.info(f"*** Episode {episode}/{num_episodes}, Durchschnittlicher Score (letzte 10): {avg_score:.2f} ***")
                # Zusätzliche Logging-Information
                logger.debug(f"Letzte 10 Scores: {scores[-10:]}")

            # Durchschnittlichen Score alle 100 Episoden berechnen
            if episode % 100 == 0:
                last_100_scores = scores[-100:]
                avg_score = sum(last_100_scores) / len(last_100_scores) if last_100_scores else 0
                avg_scores.append(avg_score)
                writer.add_scalar('Score/100_avg', avg_score, episode)  # Logge den durchschnittlichen Score
                logger.info(f"*** Episode {episode}/{num_episodes}, Durchschnittlicher Score (letzte 100): {avg_score:.2f} ***")

                # Early Stopping: Stoppe das Training, wenn sich der durchschnittliche Score nicht verbessert hat
                if len(avg_scores) > 5 and all(avg_scores[-1] <= s for s in avg_scores[-6:-1]):
                    logger.info("Keine Verbesserung der durchschnittlichen Scores in den letzten 500 Episoden. Training wird gestoppt.")
                    break

    except KeyboardInterrupt:
        logger.info("Training abgebrochen durch Benutzer.")
    except Exception as e:
        logger.error(f"Ein Fehler ist aufgetreten: {e}")
    finally:
        env.close()
        writer.close()  # Schließe TensorBoard

    # Plot der durchschnittlichen Scores, falls vorhanden
    if avg_scores:
        plt.figure(figsize=(10, 5))
        episodes_range = range(100, episode + 1, 100)
        plt.plot(episodes_range, avg_scores, marker='o', label='Durchschnittlicher Score')
        plt.xlabel('Episode')
        plt.ylabel('Durchschnittlicher Score (letzte 100)')
        plt.title('Trainingsfortschritt des DQN-Agenten')
        plt.grid(True)
        plt.legend()
        plt.savefig('average_scores.png')
        plt.show()
    else:
        logger.warning("Keine durchschnittlichen Scores vorhanden, um sie zu plotten.")

    # Plot der Trainingsverluste, falls vorhanden
    if losses:
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Trainingsverlust')
        plt.xlabel('Schritt')
        plt.ylabel('Verlust')
        plt.title('Trainingsverlust des DQN-Agenten')
        plt.grid(True)
        plt.legend()
        plt.savefig('training_losses.png')
        plt.show()
    else:
        logger.warning("Keine Verluste vorhanden, um sie zu plotten.")

    # Speichere das trainierte Modell und den Replay-Puffer
    try:
        agent.save_model(model_path)
        logger.info(f"Modell erfolgreich gespeichert unter {model_path}")

        with open(memory_path, 'wb') as f:
            pickle.dump(agent.memory, f)
        logger.info(f"Replay-Puffer erfolgreich gespeichert unter {memory_path}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Modells oder Replay-Puffers: {e}")

if __name__ == "__main__":
    train()
