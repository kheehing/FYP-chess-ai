import matplotlib.pyplot as plt
import numpy as np
from chessEnv import ChessEnv as CE
from dqnAgent import DQNAgent

env = CE()
agent = DQNAgent(state_size=(8, 8, 12), action_size=env.action_space.n)

# Training loop with saving the model
episodes = 3000
batch_size = 32
rewards = []

for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 8, 8, 12])
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        legal_moves = list(env.board.legal_moves)

        if action >= len(legal_moves):
            action = np.random.choice(len(legal_moves))

        next_state, reward, done, _ = env.step(action)
        
        next_state = np.reshape(next_state, [1, 8, 8, 12])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")
            rewards.append(total_reward)  # Track the reward

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Save the model
agent.model.save('keras-model/1000_new.keras')

# Plotting the rewards
plt.plot(range(episodes), rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress')
plt.show()
