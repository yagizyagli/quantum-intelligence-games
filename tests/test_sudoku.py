import unittest
import numpy as np
from src.sudoku.grid import QuantumSudokuGrid
from src.sudoku.solver import QuantumSudokuSolver

class TestQuantumSudokuConstraints(unittest.TestCase):
    """
    Automated unit testing suite validating the 3D amplitude tensor grid behavior
    and minimum entropy resolution algorithms.
    """

    def setUp(self):
        """Initializes testing fixtures before each distinct test case."""
        self.grid = QuantumSudokuGrid()
        self.solver = QuantumSudokuSolver(self.grid)

    def test_initial_maximum_superposition(self):
        """Validates that an empty grid initializes cell values into uniform probability density."""
        probs = self.grid.get_cell_probabilities(0, 0)
        # 1/9 probability for each number (1-9) -> approx 0.11111
        for val in probs:
            self.assertAlmostEqual(float(val), 1.0/9.0, places=5)

    def test_deterministic_collapse(self):
        """Validates that manually pinning a value clears out other state probabilities completely."""
        self.grid.set_cell_deterministic(0, 0, 5)
        probs = self.grid.get_cell_probabilities(0, 0)
        
        self.assertEqual(float(probs[4]), 1.0) # Index 4 maps to number 5
        self.assertEqual(float(np.sum(probs)), 1.0)
        self.assertEqual(int(np.count_nonzero(probs > 1e-5)), 1)

if __name__ == "__main__":
    unittest.main()
