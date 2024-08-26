import optuna
import numpy as np
from dqnAgent_hyper import DQNAgent
from chessEnv import ChessEnv
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator

def objective(trial):
    # Define hyperparameters to be tuned
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_uniform('gamma', 0.8, 0.999)
    epsilon = trial.suggest_uniform('epsilon', 0.1, 1.0)
    
    # Initialize environment and agent
    env = ChessEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, epsilon=epsilon)
    
    num_episodes = 100  
    total_reward = 0
    for e in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1] + list(state.shape))
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1] + list(next_state.shape))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            if len(agent.memory) > agent.batch_size:
                agent.replay(agent.batch_size)
    
    # The objective function returns the cumulative reward (or negative loss, depending on your metric)
    return total_reward / num_episodes


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

print(f"Best trial: {study.best_trial.params}")

evaluator = MeanDecreaseImpurityImportanceEvaluator()
optuna.visualization.plot_param_importances(study, evaluator=evaluator)