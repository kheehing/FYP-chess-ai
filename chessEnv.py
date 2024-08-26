
import gym, chess
import numpy as np
from collections import defaultdict

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = gym.spaces.Discrete(1001)  # Max moves
        self.observation_space = gym.spaces.Box(0, 1, (8, 8, 12), dtype=int)
        self.halfmove_clock = 0  # Tracks half-moves for the 50-move rule
        self.position_history = defaultdict(int)  # Tracks position occurrences for threefold repetition
        self.reward_shaping = True  # Enable reward shaping to avoid stalemates
        self.exploration_rate = 1.0  # Initial exploration rate for exploration-exploitation balance

    def reset(self):
        self.board.reset()
        self.halfmove_clock = 0
        self.position_history.clear()
        self._update_position_history()
        return self._get_obs()

    def step(self, action):
        if isinstance(action, chess.Move):
            move = action
        else:
            move = self._index_to_move(action)

        if move not in self.board.legal_moves:
            return self._get_obs(), -1, True, {}  # Illegal move

        # Save the current state before the move
        state_before = self._get_obs()

        self.board.push(move)
        self.halfmove_clock += 1
        self._update_position_history()

        done = False
        reward = 0

        # Check for game over conditions
        if self.board.is_checkmate():
            reward = 1  # Reward for winning
            done = True
        elif self.board.is_stalemate():
            reward = -0.5  # Penalize stalemates slightly
            done = True
        elif self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            reward = -1  # Heavier penalty for these drawn conditions
            done = True
        elif self.board.is_variant_draw():
            reward = -0.5
            done = True

        # Get the new state after the move
        state_after = self._get_obs()

        # Calculate additional step-level rewards
        material_reward = self.evaluate_material_gain_or_loss(state_before, state_after)

        # Combine rewards
        total_step_reward = reward + material_reward

        # Exploration-Exploitation adjustment
        self.exploration_rate = max(0.1, self.exploration_rate * 0.995)  # Gradually reduce exploration

        return state_after, total_step_reward, done, {}


    def _get_obs(self):
        # Convert the board state to a one-hot encoded 8x8x12 representation
        board_array = np.zeros((8, 8, 12), dtype=int)
        piece_map = self.board.piece_map()
        for square, piece in piece_map.items():
            piece_type = piece.piece_type - 1
            color_offset = 6 * int(piece.color)
            board_array[chess.square_rank(square), chess.square_file(square), piece_type + color_offset] = 1
        return board_array

    def _update_position_history(self):
        board_state = self.board.fen()
        self.position_history[board_state] += 1

    def _index_to_move(self, index):
        legal_moves = list(self.board.legal_moves)
        if index < len(legal_moves):
            return legal_moves[index]
        return 
    
    def get_valid_actions(self):
        return list(self.board.legal_moves)

    def evaluate_material_gain_or_loss(self, state_before, state_after):
        piece_values = {
            1: 10,  # Pawn
            2: 30,  # Knight
            3: 30,  # Bishop
            4: 50,  # Rook
            5: 90,  # Queen
            6: 100,  # King (check / checkmate)
            -1: -10,  # Opponent's Pawn
            -2: -30,  # Opponent's Knight
            -3: -30,  # Opponent's Bishop
            -4: -50,  # Opponent's Rook
            -5: -90,  # Opponent's Queen
            -6: -100   # Opponent's King (check / checkmate)
        }
        
        def material_count(state):
            total = 0
            for row in state:
                for piece in row:
                    # Ensure piece is an integer
                    if isinstance(piece, np.ndarray):
                        if piece.size == 1:
                            piece = piece.item()  # Convert single-element array to a scalar
                        else:
                            continue  # Skip non-scalar entries (in case of unexpected shapes)
                    total += piece_values.get(piece, 0)  # Get the value of the piece, defaulting to 0 if not found
            return total
        
        # Calculate material before and after the move
        material_before = material_count(state_before)
        material_after = material_count(state_after)
        
        # Positive if gain, negative if loss
        return material_after - material_before
        
    def get_result(self):
        if self.board.is_checkmate():
            return -1 if self.board.turn else 1
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return 0
        elif self.board.is_game_over():
            # If game is over but no other conditions met, return draw as fallback
            return 0
        else:
            return None