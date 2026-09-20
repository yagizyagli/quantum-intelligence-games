import numpy as np
from typing import Tuple, List, Optional
from src.chess.board import QuantumChessBoard

class QuantumChessEngine:
    """
    Handles move validation, legality checks, and classical-quantum hybrid physics 
    rules for Quantum Chess matches.
    """
    def __init__(self, board: QuantumChessBoard):
        self.board = board

    def is_classical_path_clear(self, from_coord: Tuple[int, int], to_coord: Tuple[int, int]) -> bool:
        """
        Checks if the linear path (horizontal, vertical, diagonal) between two squares 
        is completely empty in the macro state view.
        """
        fx, fy = from_coord
        tx, ty = to_coord
        
        dx = np.sign(tx - fx)
        dy = np.sign(ty - fy)
        
        curr_x, curr_y = fx + dx, fy + dy
        while (curr_x, curr_y) != (tx, ty):
            if self.board.grid[curr_x, curr_y] is not None:
                return False
            curr_x += dx
            curr_y += dy
            
        return True

    def validate_quantum_move(self, piece_name: str, from_coord: Tuple[int, int], 
                              to_coord: Tuple[int, int]) -> bool:
        """
        Validates if a move is standard or requires a quantum resolution.
        If a path contains a piece in a superposition state, it triggers a probability threshold check.
        """
        if piece_name not in self.board.piece_positions:
            return False
            
        # Basic boundary verification
        tx, ty = to_coord
        if not (0 <= tx < self.board.board_size and 0 <= ty < self.board.board_size):
            return False
            
        # Check target square occupancy probability
        probabilities = self.board.quantum_state.calculate_probabilities()
        target_prob = probabilities[tx, ty]
        
        # If target has a 100% deterministic allied piece, move is illegal
        target_piece = self.board.grid[tx, ty]
        if target_piece and target_piece[0] == piece_name[0] and target_prob == 1.0:
            return False
            
        return True

    def resolve_quantum_capture(self, attacker_name: str, target_coord: Tuple[int, int]) -> str:
        """
        Resolves a situation where an attacking piece lands on a square occupied by a 
        piece in superposition. Forces a measurement collapse on the target tile.
        Returns the winner entity string designation ('attacker' or 'ghost').
        """
        tx, ty = target_coord
        probabilities = self.board.quantum_state.calculate_probabilities()
        
        # If the piece is only partially there, trigger a measurement observation
        if 0.0 < probabilities[tx, ty] < 1.0:
            cx, cy = self.board.quantum_state.measure_and_collapse()
            
            # If wave function collapsed to this square, the target piece is actually here and captured
            if (cx, cy) == (tx, ty):
                captured_piece = self.board.grid[tx, ty]
                if captured_piece in self.board.piece_positions:
                    del self.board.piece_positions[captured_piece]
                self.board.grid[tx, ty] = attacker_name
                return "attacker_win"
            else:
                # The piece was a ghost state, attacker lands on an empty square
                self.board.grid[tx, ty] = attacker_name
                return "ghost_passed"
                
        # If target piece was 100% there deterministically
        elif probabilities[tx, ty] == 1.0:
            captured_piece = self.board.grid[tx, ty]
            if captured_piece in self.board.piece_positions:
                del self.board.piece_positions[captured_piece]
            self.board.grid[tx, ty] = attacker_name
            return "deterministic_capture"
            
        return "empty_square_move"
