import keras, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chessEnv import ChessEnv
from itertools import combinations

env = ChessEnv()

# Load the models
models = []
model_directory = "keras-model/"
for filename in os.listdir(model_directory):
    if filename.endswith('.keras'):
        model_path = os.path.join(model_directory, filename)
        model = keras.models.load_model(model_path)
        models.append(model)

def simulate_game(model_white, model_black, env):
    state = env.reset()
    state = np.reshape(state, [1, 8, 8, 12])
    current_player = 1  # 1 -> white, 2 -> black
    done = False
    fens = []  # List to store FEN strings

    while not done:
        if not env.board.is_game_over():
            fens.append(env.board.fen())  # Store the FEN notation
            
            if current_player == 1:
                action = np.argmax(model_white.predict(state, verbose=0)[0])
                print(f"action: {action}")
                print(f"White's move (action): {action}")
            else:
                action = np.argmax(model_black.predict(state, verbose=0)[0])
                print(f"action: {action}")
                print(f"Black's move (action): {action}")

            # Check if the action is valid
            try:
                next_state, _, done, _ = env.step(action)
                state = np.reshape(next_state, [1, 8, 8, 12])
                current_player = 3 - current_player
            except Exception as e:
                print(f"Error during step: {e}")
                return 0  # Return a draw if something goes wrong
        else:
            done = True

    # After the loop, we check if the game is over
    if env.board.is_game_over():
        result = env.get_result()
        if result is not None:
            print(f"Game result: {result}")
            print("Game FENs:")
            for fen in fens:
                print(fen)
            return result
        else:
            print("Error: Game ended but result is None.")
            return 0  # Default to draw if something goes wrong
    else:
        print("Unexpected state: Game did not finish properly.")
        return 0  # Default to draw in case of an unexpected state

def flip_stats(result):
    return -result

def simulate_100_games(model_a, model_b, env):
    results = []
    for i in range(10):
        if i % 2 == 0:
            results.append(simulate_game(model_a, model_b, env))
        else:
            result = simulate_game(model_b, model_a, env)
            results.append(flip_stats(result))
    return results

results_dict = {}

for i, j in combinations(range(len(models)), 2):
    model_a = models[i]
    model_b = models[j]
    
    # Simulate games between model_a and model_b
    results = simulate_100_games(model_a, model_b, env)
    
    # Calculate the statistics
    wins_a = sum(1 for result in results if result == 1)
    wins_b = sum(1 for result in results if result == -1)
    draws = sum(1 for result in results if result == 0)
    
    # Store the results
    results_dict[f"Model {i+1} vs Model {j+1}"] = (wins_a, wins_b, draws)

# Convert the results dictionary to a DataFrame for easier plotting
results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Wins A', 'Wins B', 'Draws'])

# Create a table plot for wins (both Wins A and Wins B)
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
wins_table = ax.table(cellText=results_df[['Wins A', 'Wins B']].values, 
                      colLabels=['Wins A', 'Wins B'], 
                      rowLabels=results_df.index, 
                      loc='center', cellLoc='center')
wins_table.auto_set_font_size(False)
wins_table.set_fontsize(10)
ax.set_title("Wins for Model A vs Model B")

plt.show()

# Create a table plot for draws
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
draws_table = ax.table(cellText=results_df[['Draws']].values, 
                       colLabels=['Draws'], 
                       rowLabels=results_df.index, 
                       loc='center', cellLoc='center')
draws_table.auto_set_font_size(False)
draws_table.set_fontsize(10)
ax.set_title("Draws for Model A vs Model B")

plt.show()
