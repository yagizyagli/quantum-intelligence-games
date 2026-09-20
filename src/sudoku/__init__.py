"""
Quantum Sudoku Module.
Exposes the QuantumSudokuGrid and QuantumSudokuSolver for managing 
3D value amplitude tensors and quantum constraint satisfaction heuristics.
"""

from src.sudoku.grid import QuantumSudokuGrid
from src.sudoku.solver import QuantumSudokuSolver

__all__ = [
    "QuantumSudokuGrid",
    "QuantumSudokuSolver",
]
