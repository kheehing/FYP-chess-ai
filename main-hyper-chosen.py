import matplotlib.pyplot as plt
import numpy as np
from chessEnv import ChessEnv
from dqnAgent_hyper import DQNAgent
import chess


env = ChessEnv()

learning_rate = 0.0007415301872174259
batch_size = 256
gamma = 0.9085046347570451
epsilon = 0.5541469916437142
state_size = (8, 8, 12)
action_size=env.action_space.n
agent = DQNAgent(state_size, action_size, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, epsilon=epsilon)

# Training loop with saving the model
episodes = 3000
rewards = []

for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 8, 8, 12])
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        legal_moves = env.board.legal_moves

        if action >= len(legal_moves):
            total_reward -= 1
            action = np.random.choice(len(legal_moves))
        else:
            rewards += 1 

        next_state, reward, done, _ = env.step(action)
        
        # Reward shaping logic
        material_reward = env.evaluate_material_gain_or_loss(state, next_state)
        positional_reward = env.evaluate_position_advantage(state, next_state)
        
        # Combine rewards with the environment reward (e.g., game won or lost)
        total_step_reward = reward + material_reward + positional_reward
        
        next_state = np.reshape(next_state, [1, 8, 8, 12])
        agent.remember(state, action, total_step_reward, next_state, done)
        state = next_state
        total_reward += total_step_reward

        if done:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")
            rewards.append(total_reward)  # Track the reward

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Save the model
agent.model.save('keras-model/3000.keras')

# Plotting the rewards
plt.plot(range(episodes), rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress')
plt.show()
