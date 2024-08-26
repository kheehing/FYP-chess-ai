
import os
import numpy as np
from chessEnv import ChessEnv
from dqnAgent import DQNAgent

# Parameters
population_size = 15
generations = 50
episodes_per_agent = 100
batch_size = 32
mutation_rate = 0.1
save_folder = 'evolution-1'

# Create the save directory if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Initialize population
env = ChessEnv()
population = [DQNAgent(state_size=(8, 8, 12), action_size=env.action_space.n) for _ in range(population_size)]

def mutate_agent(agent):
    """Apply random mutations to the agent's neural network weights."""
    weights = agent.model.get_weights()
    new_weights = []
    for weight in weights:
        if np.random.rand() < mutation_rate:
            new_weights.append(weight + np.random.normal(0, 0.1, weight.shape))
        else:
            new_weights.append(weight)
    agent.model.set_weights(new_weights)

def evaluate_agent(agent):
    """Evaluate the agent's performance over a number of episodes."""
    total_reward = 0
    for _ in range(episodes_per_agent):
        state = env.reset()
        state = np.reshape(state, [1, 8, 8, 12])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 8, 8, 12])
            total_reward += reward
            state = next_state
    return total_reward

def evolve_population(population):
    """Evolve the population by selecting the top performers and reproducing."""
    # Evaluate all agents
    scores = [(agent, evaluate_agent(agent)) for agent in population]
    scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending

    # Select the top half of the population
    top_half = [agent for agent, _ in scores[:population_size // 2]]

    # Reproduce and mutate to create a new population
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = np.random.choice(top_half, 2, replace=False)
        child = DQNAgent(state_size=(8, 8, 12), action_size=env.action_space.n)
        
        # Average the weights layer by layer
        child_weights = []
        for w1, w2 in zip(parent1.model.get_weights(), parent2.model.get_weights()):
            child_weights.append((w1 + w2) / 2)
        
        child.model.set_weights(child_weights)
        mutate_agent(child)
        new_population.append(child)

    return new_population


# Evolution loop
for generation in range(generations):
    print(f"Generation {generation + 1}/{generations}")

    # Evolve the population
    population = evolve_population(population)

    # Save each agent in the population
    for idx, agent in enumerate(population):
        agent.model.save(os.path.join(save_folder, f"agent_gen{generation+1}_id{idx+1}.h5"))

    print(f"Generation {generation + 1} completed and saved.")

print("Evolution completed.")
