import keras, os
import numpy as np
import matplotlib.pyplot as plt
from chessEnv import ChessEnv
from itertools import combinations

# Load Environment
env = ChessEnv()

# Load the models
models = []
model_directory = "keras-model/"
for filename in os.listdir(model_directory):
    if filename.endswith('.keras'):
        model_path = os.path.join(model_directory, filename)
        model = keras.models.load_model(model_path)
        models.append((model, filename))

def simulate_game(model_white, model_black, env):
    state = env.reset()
    state = np.reshape(state, [1, 8, 8, 12])
    current_player = 1  # 1 -> white, 2 -> black
    done = False
    fens = []  # List to store FEN strings for the game

    while not done:
        if not env.board.is_game_over():
            fens.append(env.board.fen())  # Store the FEN string for the current board state
            print(env.board)  # Print the board state for debugging
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
                return 0, fens  # Return a draw and the FENs if something goes wrong
        else:
            done = True

    # After the loop, we check if the game is over
    if env.board.is_game_over():
        result = env.get_result()
        if result is not None:
            print(f"Game result: {result}")
            return result, fens  # Return the result and FENs
        else:
            print("Error: Game ended but result is None.")
            return 0, fens  # Default to draw and FENs if something goes wrong
    else:
        print("Unexpected state: Game did not finish properly.")
        return 0, fens  # Default to draw and FENs in case of an unexpected state

# Simulate games
results_dict = {}
fens_dict = {}

for i, j in combinations(range(len(models)), 2):
    model_a, filename_a = models[i]
    model_b, filename_b = models[j]
    
    wins_a = 0
    wins_b = 0
    draws = 0
    all_fens = []  # List to store all FENs for all games between these two models
    
    # Simulate multiple games
    for game_num in range(10):
        result, fens = simulate_game(model_a, model_b, env)
        
        # Accumulate the FENs
        all_fens.extend(fens)
        
        # Calculate the game result
        if result == 1:
            wins_a += 1
        elif result == -1:
            wins_b += 1
        else:
            draws += 1
    
    # Store the results using filenames
    results_dict[f"{filename_a} vs {filename_b}"] = (wins_a, wins_b, draws)
    
    # Store all FENs for the games between these two models
    fens_dict[f"{filename_a} vs {filename_b}"] = all_fens


# display results
def display_results_and_summary(results_dict, fens_dict):
    summary = {}

    # Iterate through each matchup and display its results in a separate figure
    for matchup, result in results_dict.items():
        wins_a, wins_b, draws = result
        
        # Store the summary for later display
        summary[matchup] = {
            "Wins A": wins_a,
            "Wins B": wins_b,
            "Draws": draws,
            "Total Games": wins_a + wins_b + draws
        }
        
        # Create a new figure for each matchup
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Wins A', 'Wins B', 'Draws'], [wins_a, wins_b, draws], color=['blue', 'red', 'gray'])
        ax.set_title(f"Results for {matchup}")
        ax.set_ylabel("Number of Games")
        ax.set_ylim(0, max(wins_a, wins_b, draws) + 1)
        
        # Show the plot for this matchup
        plt.tight_layout()
        plt.show()
        
        # Print FENs to the terminal for this matchup
        print(f"FENs for {matchup}:")
        for fen in fens_dict[matchup]:
            print(fen)
        print("\n" + "-"*40 + "\n")  # Separator between matchups for clarity

    # Print the summary of results
    print("\nSummary of Results:\n")
    for matchup, stats in summary.items():
        print(f"{matchup}:")
        print(f"  Wins A: {stats['Wins A']}")
        print(f"  Wins B: {stats['Wins B']}")
        print(f"  Draws: {stats['Draws']}")
        print(f"  Total Games: {stats['Total Games']}")
        print("\n" + "-"*40 + "\n")

# Call the function to display results, summary, and print FENs in the terminal
display_results_and_summary(results_dict, fens_dict)