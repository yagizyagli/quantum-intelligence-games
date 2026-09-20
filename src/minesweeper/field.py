import numpy as np
from typing import Tuple, Dict, List, Set
from src.core.state import QuantumState

class QuantumMinesweeperField:
    """
    Manages a Quantum Minesweeper Field where cells contain mine probabilities 
    in a state of superposition until revealed (measured) by the player.
    """
    def __init__(self, size: int = 10, mine_density: float = 0.2):
        self.size = size
        self.mine_density = mine_density
        self.quantum_state = QuantumState(size=self.size)
        
        # Set tracking which coordinates have been explicitly revealed/collapsed
        self.revealed_cells: Set[Tuple[int, int]] = set()
        
        # Grid containing the absolute classical state post-collapse (0: Clear, 1: Mine)
        self.classical_grid = np.zeros((size, size), dtype=int)
        self._initialize_probabilistic_mines()

    def _initialize_probabilistic_mines(self) -> None:
        """Fills the quantum state matrix with complex amplitudes based on target mine density."""
        # Calculate amplitude for the target density: sqrt(density) -> |amplitude|^2 = density
        target_amplitude = np.sqrt(self.mine_density)
        
        for r in range(self.size):
            for c in range(self.size):
                # Apply complex amplitude representing the probabilistic superposition of a mine
                self.quantum_state.apply_superposition(r, c, target_amplitude + 0j)

    def get_cell_mine_probability(self, row: int, col: int) -> float:
        """Returns the current calculated probability of a mine being at the specified coordinate."""
        probabilities = self.quantum_state.calculate_probabilities()
        return float(probabilities[row, col])

    def reveal_cell(self, row: int, col: int) -> Tuple[bool, int]:
        """
        Triggers a measurement observation on a targeted cell. Collapses the local wave function.
        Returns a tuple: (is_mine_detonated: bool, neighboring_mines_count: int)
        """
        if (row, col) in self.revealed_cells:
            return False, self._count_neighboring_mines(row, col)

        # Calculate live probability map prior to observation
        prob = self.get_cell_mine_probability(row, col)
        
        # Collapse event simulation for this precise cell based on its live probability amplitude
        is_mine = np.random.choice([1, 0], p=[prob, 1.0 - prob])
        self.classical_grid[row, col] = is_mine
        self.revealed_cells.add((row, col))
        
        # If it collapses into a deterministic mine, adjust its quantum state amplitude to 100%
        if is_mine == 1:
            self.quantum_state.apply_superposition(row, col, 1.0 + 0j)
            return True, 0
        else:
            self.quantum_state.apply_superposition(row, col, 0j)
            
        return False, self._count_neighboring_mines(row, col)

    def _count_neighboring_mines(self, row: int, col: int) -> int:
        """Counts how many collapsed deterministic mines or probable fields exist in adjacent cells."""
        mine_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    # If neighbor is already revealed, look at deterministic grid
                    if (nr, nc) in self.revealed_cells:
                        mine_count += self.classical_grid[nr, nc]
                    else:
                        # If unrevealed, count statistically if probability passes an empirical threshold
                        if self.get_cell_mine_probability(nr, nc) > 0.5:
                            mine_count += 1
        return mine_count
