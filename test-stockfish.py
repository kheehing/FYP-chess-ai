import chess
import chess.engine

# Initialize Stockfish with different depths or skill levels
stockfish_1000 = chess.engine.SimpleEngine.popen_uci("C:/Users/light/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")
stockfish_2000 = chess.engine.SimpleEngine.popen_uci("C:/Users/light/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")
stockfish_3000 = chess.engine.SimpleEngine.popen_uci("C:/Users/light/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")

stockfish_1000.configure({"Skill Level": 10, "Threads": 2, "Hash": 256})
stockfish_2000.configure({"Skill Level": 15, "Threads": 2, "Hash": 256})
stockfish_3000.configure({"Skill Level": 20, "Threads": 2, "Hash": 256})

# Function to run a game between two engines
def play_game(engine_white, engine_black):
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            result = engine_white.play(board, chess.engine.Limit(time=0.1))
        else:
            result = engine_black.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
    return board.result()

# multiple games between two engines swaping black and white
def compete(engine1, engine2, games=1000):
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    for _ in range(games):
        # engine1:White      engine2:Black
        result = play_game(engine1, engine2)
        results[result] += 1
        
        # engine1:Black      engine2:White
        result = play_game(engine2, engine1)
        results[result] += 1
    
    return results

# Run competitions
results_1000_vs_2000 = compete(stockfish_1000, stockfish_2000, games=50)
results_2000_vs_3000 = compete(stockfish_2000, stockfish_3000, games=50)
results_1000_vs_3000 = compete(stockfish_1000, stockfish_3000, games=50)

# Print results
print("Results for Stockfish 1000 vs Stockfish 2000:", results_1000_vs_2000)
print("Results for Stockfish 2000 vs Stockfish 3000:", results_2000_vs_3000)
print("Results for Stockfish 1000 vs Stockfish 3000:", results_1000_vs_3000)

# Close engines
stockfish_1000.quit()
stockfish_2000.quit()
stockfish_3000.quit()
