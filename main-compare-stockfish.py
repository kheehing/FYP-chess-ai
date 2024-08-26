import chess
import chess.engine
import numpy as np
import keras
from stockfish import Stockfish
import matplotlib.pyplot as plt

# Loading Keras model
model1 = keras.models.load_model('keras-model/1000.keras')
model2 = keras.models.load_model('keras-model/2000.keras')
model3 = keras.models.load_model('keras-model/3000.keras')

# Initialize Stockfish with the path to the Stockfish binary
stockfish = Stockfish(path="C:/Users/light/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe", depth=15)

try:
    best_move = stockfish.get_best_move()
    print(f"Best move: {best_move}")
except Exception as e:
    print(f"Error: {e}")

def get_stockfish_move(board):
    stockfish.set_fen_position(board.fen())
    return chess.Move.from_uci(stockfish.get_best_move())

def get_model_move(board, model):
    try:
        stockfish.set_fen_position(board.fen())
        best_move = stockfish.get_best_move()
        if best_move:
            return chess.Move.from_uci(best_move)
        else:
            raise Exception("Stockfish failed to return a move")
    except Exception as e:
        print(f"Error: {e}")
        return chess.Move.null()  # Return a null move in case of failure

def board_to_input(board):
    # Initialize a 8x8x12 numpy array with zeros
    input_array = np.zeros((8, 8, 12), dtype=np.float32)
    
    # Define piece mappings (for example: plane 0-5 for white pieces, 6-11 for black pieces)
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    # Loop through the board squares and populate the input array
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            piece_type = piece_map[piece.piece_type]
            color_offset = 6 if piece.color == chess.BLACK else 0
            input_array[row, col, piece_type + color_offset] = 1.0
    
    return input_array

def move_index_to_chess_move(move_index, board):
    # Assuming move_index is in the range [0, 4095], representing 64*64 possible moves
    start_square = move_index // 64  # Integer division to get the start square
    end_square = move_index % 64  # Modulo to get the end square
    
    move = chess.Move(start_square, end_square)
    
    # Ensure the move is legal
    if move in board.legal_moves:
        return move
    else:
        # Handle illegal moves (this depends on how you want to handle them)
        # You could choose to return a random legal move, or handle it differently
        return chess.Move.null()  # Return a null move if the move is illegal

def play_game(model, stockfish):
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # Your model plays as white
            move = get_model_move(board, model)
        else:
            # Stockfish plays as black
            move = get_stockfish_move(board)
        
        board.push(move)

    return board.result()

# Test the models against Stockfish at different ELO levels
models = [model1, model2, model3]
model_names = ["Model 1", "Model 2", "Model 3"]
elo_ratings = [1000, 1500, 2000]
results = {
    "Model 1": {1000: {"win": 0, "loss": 0, "draw": 0},
                1500: {"win": 0, "loss": 0, "draw": 0},
                2000: {"win": 0, "loss": 0, "draw": 0}},
    "Model 2": {1000: {"win": 0, "loss": 0, "draw": 0},
                1500: {"win": 0, "loss": 0, "draw": 0},
                2000: {"win": 0, "loss": 0, "draw": 0}},
    "Model 3": {1000: {"win": 0, "loss": 0, "draw": 0},
                1500: {"win": 0, "loss": 0, "draw": 0},
                2000: {"win": 0, "loss": 0, "draw": 0}},
}

for model, model_name in zip(models, model_names):
    for elo in elo_ratings:
        stockfish.set_elo_rating(elo)
        if elo == 1000:
            stockfish._set_option("Threads", 1) 
            stockfish._set_option("Hash", 128) 
        elif elo == 1500:
            stockfish._set_option("Threads", 2) 
            stockfish._set_option("Hash", 256) 
        elif elo == 2000:
            stockfish._set_option("Threads", 4) 
            stockfish._set_option("Hash", 512) 

        print(f"Testing {model_name} against Stockfish with ELO {elo}")
        for _ in range(100):  # Play 10 games per ELO rating
            result = play_game(model, stockfish)
            if result == "1-0":
                results[model_name][elo]["win"] += 1
            elif result == "0-1":
                results[model_name][elo]["loss"] += 1
            else:
                results[model_name][elo]["draw"] += 1

print("done testing")

# Create a bar chart to visualize the results
categories = ["Win", "Loss", "Draw"]
x = np.arange(len(categories))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 7))

# Plot bars for each model at each ELO rating
for i, model_name in enumerate(model_names):
    data_1000 = [results[model_name][1000]["win"], results[model_name][1000]["loss"], results[model_name][1000]["draw"]]
    data_1500 = [results[model_name][1500]["win"], results[model_name][1500]["loss"], results[model_name][1500]["draw"]]
    data_2000 = [results[model_name][2000]["win"], results[model_name][2000]["loss"], results[model_name][2000]["draw"]]
    
    ax.bar(x + (i - 1) * width, data_1000, width, label=f'{model_name} vs 1000 ELO')
    ax.bar(x + (i - 1) * width + width * len(models), data_1500, width, label=f'{model_name} vs 1500 ELO')
    ax.bar(x + (i - 1) * width + 2 * width * len(models), data_2000, width, label=f'{model_name} vs 2000 ELO')

# Add labels, title, and x-axis ticks
ax.set_xlabel('Result')
ax.set_ylabel('Number of Games')
ax.set_title('Model Performance Against Stockfish at Different ELO Ratings')
ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels(categories)
ax.legend()

fig.tight_layout()
plt.show()

# Print results
for model_name in model_names:
    for elo in elo_ratings:
        print(f"Results for {model_name} against Stockfish at ELO {elo}:")
        print(f"Wins: {results[model_name][elo]['win']}, Losses: {results[model_name][elo]['loss']}, Draws: {results[model_name][elo]['draw']}")
