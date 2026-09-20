import unittest
import numpy as np
from src.chess.board import QuantumChessBoard
from src.chess.engine import QuantumChessEngine

class TestQuantumChessRules(unittest.TestCase):
    """
    Automated unit testing suite validating movement mechanics, splits, merges, 
    and quantum capture resolutions on the 8x8 board.
    """

    def setUp(self):
        """Initializes testing fixtures before each distinct test case."""
        self.board = QuantumChessBoard()
        self.engine = QuantumChessEngine(self.board)

    def test_quantum_split_move(self):
        """Validates that a split move spreads a piece into an equal 50/50 superposition."""
        # Split a white pawn from its classical position (1, 0) to (2, 0) and (3, 0)
        piece = "wP1"
        self.board.execute_split_move(piece, (1, 0), (2, 0), (3, 0))

        # Check tracking dictionaries
        self.assertIn((2, 0), self.board.piece_positions[piece])
        self.assertIn((3, 0), self.board.piece_positions[piece])

        # Validate math amplitude normalization
        probabilities = self.board.quantum_state.calculate_probabilities()
        self.assertAlmostEqual(float(probabilities[2, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(probabilities[3, 0]), 0.5, places=5)

    def test_quantum_capture_resolution(self):
        """Validates that attacking a superposition tile triggers proper measurement metrics."""
        # Split white pawn to create a ghost state at (2, 2)
        self.board.execute_split_move("wP3", (1, 2), (2, 2), (3, 2))
        
        # Validate capturing response
        outcome = self.engine.resolve_quantum_capture("bN1", (2, 2))
        
        # Outcome must either collapse as an attacker win or ghost bypass
        self.assertIn(outcome, ["attacker_win", "ghost_passed"])

if __name__ == "__main__":
    unittest.main()
