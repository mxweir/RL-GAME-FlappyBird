# Flappy Bird Reinforcement Learning Project

This project is an implementation of the classic Flappy Bird game with an AI agent using Q-Learning. The agent learns how to play Flappy Bird through reinforcement learning by adjusting its actions based on rewards and penalties.

## Project Structure

- **agent.py**: Contains the implementation of the Q-Learning agent that makes decisions based on a Q-table. The agent learns through exploration and exploitation and adjusts its Q-values using rewards for navigating the bird through the gaps between pipes.
- **main.py**: The main game loop where the Flappy Bird game is run. It manages the game flow, interactions between the bird, pipes, and agent, and tracks the score.
- **game.py**: Contains the classes for the Bird and Pipe, as well as the collision detection. This file defines how the bird moves and interacts with the pipes.
- **graphics.py**: Handles the visual rendering of the game, including the bird, pipes, score, and game over screens.

## How It Works

The AI agent is trained using a Q-Learning algorithm without neural networks. The agent learns to navigate the bird through gaps by updating a Q-table based on state transitions. The state of the game is discretized into a manageable number of possible states that represent the bird's position, the distance to the next pipe, and the position of the gap.

### Reward Structure

- **Staying Alive**: The agent receives a small positive reward for each step it stays alive.
- **Moving Towards the Gap**: The agent is rewarded more when it moves towards and stays in the center of the gap between the pipes.
- **Passing Pipes**: A significant positive reward is given for successfully passing through a pipe.
- **Collisions**: The agent receives a large negative reward when it collides with a pipe or goes out of bounds, encouraging it to avoid such actions.
- **Position Relative to Gap**: The agent receives extra rewards based on its position relative to the gap center, to encourage it to align itself properly for passing through.

### Saving Progress

The Q-Learning agent periodically saves its Q-table to ensure that learning progress is not lost. The model is saved every 25 rounds, allowing the agent to pick up learning from where it left off.

## Requirements

- Python 3.x
- Pygame
- NumPy

Install the requirements using:

```
pip install pygame numpy
```

## Running the Game

To run the game, simply execute the `main.py` file:

```
python main.py
```

The game will start, and the AI agent will begin playing automatically. The Q-table will be saved periodically, and you can also manually load it to resume training from a previous point.

## Future Improvements

- **Deep Reinforcement Learning**: Replacing the Q-Learning agent with a Deep Q-Network (DQN) to handle larger state spaces more effectively.
- **Improved State Representation**: Use a more continuous state representation to improve the agent's decision-making ability.
- **Hyperparameter Tuning**: Experiment with different learning rates, discount factors, and exploration strategies to further improve the performance of the agent.

## Contributions

Feel free to fork this repository and submit pull requests with improvements or bug fixes. Contributions are welcome!

## License

This project is open-source and available under the MIT License.

