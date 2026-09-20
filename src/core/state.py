import numpy as np
from typing import Tuple

class QuantumState:
    """
    Manages quantum state amplitudes, superpositions, and measurement collapses 
    for modular board games using highly optimized matrix operations.
    """
    def __init__(self, size: int):
        self.size = size
        # Initialize state matrix with complex amplitudes representing probability amplitudes
        self.state_matrix = np.zeros((size, size), dtype=complex)

    def apply_superposition(self, x: int, y: int, amplitude: complex) -> None:
        """
        Assigns a specific complex probability amplitude to a given coordinate.
        """
        if 0 <= x < self.size and 0 <= y < self.size:
            self.state_matrix[x, y] = amplitude
        else:
            raise ValueError("Coordinates are out of bounds for the current board matrix.")

    def calculate_probabilities(self) -> np.ndarray:
        """
        Computes the absolute probability distribution (|psi|^2) across the entire board.
        Normalizes the output matrix to ensure total probability equals 1.0.
        """
        probabilities = np.abs(self.state_matrix) ** 2
        total_prob = np.sum(probabilities)
        
        if total_prob > 0:
            probabilities /= total_prob
        return probabilities

    def measure_and_collapse(self) -> Tuple[int, int]:
        """
        Observes the system, forcing the quantum wave function to collapse into a single state
        based on the probability distribution. Resets the matrix to a deterministic state.
        """
        probabilities = self.calculate_probabilities().flatten()
        
        if np.sum(probabilities) == 0:
            return (0, 0)
        
        # Perform probabilistic choice based on the state vector probabilities
        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        x, y = divmod(chosen_index, self.size)
        
        # Collapse the entire wave function into the observed state vector
        self.state_matrix = np.zeros((self.size, self.size), dtype=complex)
        self.state_matrix[x, y] = 1.0 + 0j
        
        return x, y
