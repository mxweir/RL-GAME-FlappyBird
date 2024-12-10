# test_env.py
from flappy_bird_env import FlappyBirdEnv

env = FlappyBirdEnv(render_mode=True)
state = env.reset()
done = False
steps = 0

while not done:
    action = env.action_space.sample()  # Zufällige Aktion
    next_state, reward, done, info = env.step(action)
    env.render()
    steps += 1
    print(f"Schritt: {steps}, Aktion: {action}, Belohnung: {reward}, Score: {info['score']}")

env.close()
