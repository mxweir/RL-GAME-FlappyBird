# train_dqn.py
import os
import pickle
import matplotlib.pyplot as plt
from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import torch
import time
import logging

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
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
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
    max_steps = 1000
    scores = []
    avg_scores = []
    losses = []

    start_time = time.time()

    try:
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            total_reward = 0
            episode_start_time = time.time()

            for step in range(max_steps):
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.update()  # Verlust zurückgeben
                if loss is not None:
                    losses.append(loss)
                state = next_state
                total_reward += reward

                if done:
                    break

            scores.append(info['score'])

            # Debugging-Informationen alle 10 Episoden
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                last_100_losses = losses[-100:]
                if last_100_losses:
                    avg_loss = sum(last_100_losses) / len(last_100_losses)
                else:
                    avg_loss = 0
                logger.info(f"Episode {episode}/{num_episodes}, Score: {info['score']}, Epsilon: {agent.epsilon:.4f}, Loss: {avg_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s")

            # Durchschnittlichen Score alle 100 Episoden berechnen
            if episode % 100 == 0:
                last_100_scores = scores[-100:]
                avg_score = sum(last_100_scores) / len(last_100_scores) if last_100_scores else 0
                avg_scores.append(avg_score)
                logger.info(f"*** Episode {episode}/{num_episodes}, Durchschnittlicher Score (letzte 100): {avg_score:.2f} ***")

            # Optional: Render nur bei den letzten 10 Episoden
            if episode > num_episodes - 10:
                env.render()

    except KeyboardInterrupt:
        logger.info("Training abgebrochen durch Benutzer.")
    except Exception as e:
        logger.error(f"Ein Fehler ist aufgetreten: {e}")
    finally:
        env.close()

    # Plot der durchschnittlichen Scores, falls vorhanden
    if avg_scores:
        plt.figure(figsize=(10, 5))
        episodes_range = range(100, num_episodes + 1, 100)
        plt.plot(episodes_range, avg_scores, marker='o', label='Durchschnittlicher Score')
        plt.xlabel('Episode')
        plt.ylabel('Durchschnittlicher Score (letzte 100)')
        plt.title('Trainingsfortschritt des DQN-Agenten')
        plt.grid(True)
        plt.legend()
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
        plt.show()
    else:
        logger.warning("Keine Verluste vorhanden, um sie zu plotten.")

    # Speichere das trainierte Modell und den Replay-Puffer
    try:
        if torch.cuda.is_available():
            torch.save(agent.policy_net.state_dict(), 'dqn_flappy_bird.pth')
        else:
            torch.save(agent.policy_net.state_dict(), 'dqn_flappy_bird_cpu.pth')
        logger.info(f"Modell erfolgreich gespeichert unter {model_path}")

        with open(memory_path, 'wb') as f:
            pickle.dump(agent.memory, f)
        logger.info(f"Replay-Puffer erfolgreich gespeichert unter {memory_path}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Modells oder Replay-Puffers: {e}")

    # Optional: Visualisiere einige Episoden nach dem Training
    # Aktiviert das Rendern, um den Agenten zu beobachten
    env.render_mode = True
    for _ in range(5):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            env.render()
            time.sleep(0.02)  # Pause, um das Rendering sichtbar zu machen
    env.close()

if __name__ == "__main__":
    train()
