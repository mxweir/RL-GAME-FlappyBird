# 2048 Game with Reinforcement Learning Agent

## Overview
This project is an implementation of the popular game 2048, featuring a reinforcement learning agent that can automatically learn to play and improve its performance. The game can be played manually or with the AI agent, which utilizes Q-learning to achieve higher scores. The interface combines both Tkinter for model management and Pygame for the game display.

## Features
- **Manual Play**: Play the game 2048 manually using arrow keys.
- **Reinforcement Learning Agent**: The agent learns to play the game using the Q-learning algorithm.
- **GUI for Model Management**: Manage training models (create, load, or save) using a modern Tkinter GUI.
- **Visual Feedback**: Real-time visualization of the agent's progress, including cumulative rewards, moves made, and current score.
- **Custom Window Icon**: Custom icons have been added for the window, enhancing the visual experience.

## Installation
### Requirements
To run the project, you need the following dependencies:

- Python 3.8+
- `pygame==2.1.2`
- `numpy==1.24.2`
- `ttkbootstrap==1.6.0`
- `pickle-mixin==1.0`

To install the requirements, run:
```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/2048-rl-agent.git
   cd 2048-rl-agent
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the game:
   ```bash
   python main.py
   ```

## Usage
- **Manual Mode**: Use the arrow keys (`UP`, `DOWN`, `LEFT`, `RIGHT`) to move the tiles manually.
- **Agent Mode**: Press `A` to toggle the agent. The agent will start learning and playing automatically.
- **Restart Game**: If the game is over, press `R` to restart.
- **Model Management**:
  - A GUI will appear at the beginning to load or create a new model.
  - Use sliders to adjust the `epsilon` parameter while creating new models.

## File Structure
- `main.py`: The entry point of the project that initializes the game and manages agent training.
- `agent.py`: Contains the Q-learning logic for the reinforcement learning agent.
- `game.py`: Handles the game logic for 2048, including board updates and score calculation.
- `graphics.py`: Manages the graphical interface for the game, including drawing the board and agent information.
- `utils/favicon.ico`: Custom icon used for both the Pygame and Tkinter windows.

## Agent Training Details
The Q-learning agent uses an epsilon-greedy approach to balance exploration and exploitation. You can dynamically adjust the `epsilon` value using the GUI when creating new models, which helps fine-tune the learning process.

The agent saves the learned `Q-table` automatically after every game over, allowing future reuse and continuous improvement.

## Future Improvements
- **Deep Q-Learning**: Upgrade the agent to use a neural network for approximating Q-values.
- **Enhanced GUI**: Improve the interface to provide better feedback on agent performance.
- **Optimized Training**: Add more advanced training techniques like experience replay or prioritized sampling.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The game is inspired by the classic 2048 game created by Gabriele Cirulli.
- Reinforcement learning concepts were applied to automate the gameplay and improve performance over time.

## Contributions
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

If you have any questions or feedback, feel free to reach out.

Happy Coding!

