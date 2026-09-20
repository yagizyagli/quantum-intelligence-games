import numpy as np
from typing import Dict, List, Tuple

class QuantumSudokuGrid:
    """
    Represents a 9x9 Quantum Sudoku Grid where cells can hold multiple numbers 
    simultaneously in a state of quantum superposition until observed.
    """
    def __init__(self):
        self.grid_size = 9
        
        # 3D Tensor to store probability amplitudes for each value (1-9) per cell.
        # Shape: (9, 9, 9) -> Row, Column, Value Index (0 maps to number 1, 8 maps to number 9)
        self.amplitude_tensor = np.zeros((9, 9, 9), dtype=complex)
        self._initialize_empty_superpositions()

    def _initialize_empty_superpositions(self) -> None:
        """Initializes all cells in an even, maximum superposition of all numbers 1-9."""
        # Equal probability amplitude for 9 states is 1 / sqrt(9) = 1 / 3
        equal_amplitude = 1.0 / 3.0
        self.amplitude_tensor.fill(equal_amplitude + 0j)

    def set_cell_deterministic(self, row: int, col: int, value: int) -> None:
        """
        Collapses a specific cell manually to a deterministic value (100% probability),
        clearing out all other superpositions in that specific cell.
        """
        if not (1 <= value <= 9):
            raise ValueError("Sudoku values must be between 1 and 9.")
            
        self.amplitude_tensor[row, col, :] = 0j
        self.amplitude_tensor[row, col, value - 1] = 1.0 + 0j

    def get_cell_probabilities(self, row: int, col: int) -> np.ndarray:
        """
        Calculates the absolute probability vector (|psi|^2) for numbers 1-9 in a specific cell.
        """
        amplitudes = self.amplitude_tensor[row, col, :]
        probabilities = np.abs(amplitudes) ** 2
        
        total_prob = np.sum(probabilities)
        if total_prob > 0:
            probabilities /= total_prob
            
        return probabilities

    def observe_and_collapse_cell(self, row: int, col: int) -> int:
        """
        Forces a measurement on a specific coordinate. The cell's wave function collapses
        into a single classical number based on its internal probability distribution.
        """
        probabilities = self.get_cell_probabilities(row, col)
        
        # Pick a value based on the quantum probability distribution
        chosen_index = np.random.choice(9, p=probabilities)
        chosen_value = chosen_index + 1
        
        # Fix the state deterministically post-measurement
        self.set_cell_deterministic(row, col, chosen_value)
        return chosen_value

    def verify_quantum_constraints(self, row: int, col: int, value: int) -> float:
        """
        Checks the total conflicting probability amplitude existing in the same 
        row, column, or 3x3 sub-grid for a target value. Returns a conflict score.
        """
        val_idx = value - 1
        conflict_score = 0.0
        
        # Row and Column check (summing up conflict probabilities)
        for i in range(self.grid_size):
            if i != col:
                conflict_score += np.abs(self.amplitude_tensor[row, i, val_idx]) ** 2
            if i != row:
                conflict_score += np.abs(self.amplitude_tensor[i, col, val_idx]) ** 2
                
        # 3x3 Sub-grid box check
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if r != row and c != col:
                    conflict_score += np.abs(self.amplitude_tensor[r, c, val_idx]) ** 2
                    
        return conflict_score
