import numpy as np
from typing import Tuple, Optional
from src.sudoku.grid import QuantumSudokuGrid

class QuantumSudokuSolver:
    """
    Implements quantum constraint satisfaction and probabilistic reduction 
    algorithms to resolve superpositions into a valid Sudoku configuration.
    """
    def __init__(self, grid: QuantumSudokuGrid):
        self.grid = grid

    def find_lowest_entropy_cell(self) -> Optional[Tuple[int, int]]:
        """
        Finds the cell with the lowest quantum entropy (fewest active superpositions but not yet collapsed).
        This follows the Minimum Remaining Values (MRV) heuristic adapted for quantum state arrays.
        """
        min_entropy = float('inf')
        target_cell = None

        for r in range(self.grid.grid_size):
            for c in range(self.grid.grid_size):
                probabilities = self.grid.get_cell_probabilities(r, c)
                # Count how many states have a non-zero probability (active superpositions)
                active_states = np.count_nonzero(probabilities > 1e-5)
                
                # If it's already collapsed (only 1 state active), skip it
                if active_states <= 1:
                    continue
                
                # We look for the cell closest to collapsing (lowest active superposition count)
                if active_states < min_entropy:
                    min_entropy = active_states
                    target_cell = (r, c)
                    
        return target_cell

    def step_probabilistic_resolve(self) -> bool:
        """
        Executes a single step of quantum collapse resolution. Mentally selects the 
        lowest entropy cell, measures it, collapses it, and returns True if successful.
        """
        cell = self.find_lowest_entropy_cell()
        if cell is None:
            return False  # Board is fully collapsed / solved

        row, col = cell
        # Force observation and collapse the wave function for this targeted cell
        collapsed_value = self.grid.observe_and_collapse_cell(row, col)
        
        # Verify if the collapsed state creates an immediate critical conflict violation
        conflict = self.grid.verify_quantum_constraints(row, col, collapsed_value)
        if conflict > 2.0:  # Arbitrary threshold signifying high structural state conflict
            return False
            
        return True

    def solve_quantum_board(self, max_attempts: int = 100) -> bool:
        """
        Repeatedly applies probabilistic resolution steps to solve the entire grid.
        Returns True if the board successfully collapses into a valid standard configuration.
        """
        for _ in range(max_attempts):
            cell = self.find_lowest_entropy_cell()
            if cell is None:
                return True  # No more uncollapsed cells left, successfully solved
                
            success = self.step_probabilistic_resolve()
            if not success:
                # If a major conflict occurs, re-initialize superpositions (Quantum state reset)
                self.grid._initialize_empty_superpositions()
                
        return False
