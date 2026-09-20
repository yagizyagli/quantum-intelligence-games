import numpy as np
from typing import Dict, List, Tuple, Optional
from src.core.state import QuantumState
from src.core.entanglement import EntanglementManager

class QuantumChessBoard:
    """
    Represents a Quantum Chess Board (8x8) where pieces can exist in superpositions 
    and become entangled through split and merge mechanics.
    """
    def __init__(self):
        self.board_size = 8
        self.quantum_state = QuantumState(size=self.board_size)
        self.entanglement_manager = EntanglementManager()
        
        # Maps a unique piece identifier string (e.g., 'wQ' for white Queen, 'bP1' for black Pawn 1)
        # to a list of coordinates where it currently exists in superposition.
        self.piece_positions: Dict[str, List[Tuple[int, int]]] = {}
        
        # Grid array mapping coordinates to the piece name currently occupying it.
        self.grid: np.ndarray = np.empty((8, 8), dtype=object)
        self._initialize_classical_pieces()

    def _initialize_classical_pieces(self) -> None:
        """Initializes standard chess pieces in a deterministic classical configuration."""
        back_row = ['R1', 'N1', 'B1', 'Q', 'K', 'B2', 'N2', 'R2']
        
        # Setup white and black back rows and pawns
        for i, piece in enumerate(back_row):
            self._set_piece_deterministically(f"w{piece}", 0, i)
            self._set_piece_deterministically(f"b{piece}", 7, i)
            self._set_piece_deterministically(f"wP{i+1}", 1, i)
            self._set_piece_deterministically(f"bP{i+1}", 6, i)

    def _set_piece_deterministically(self, piece_name: str, x: int, y: int) -> None:
        """Helper to set a piece completely onto a single square (Classical limit)."""
        self.grid[x, y] = piece_name
        self.piece_positions[piece_name] = [(x, y)]
        self.quantum_state.apply_superposition(x, y, 1.0 + 0j)

    def execute_split_move(self, piece_name: str, from_coord: Tuple[int, int], 
                           to_coord_a: Tuple[int, int], to_coord_b: Tuple[int, int]) -> None:
        """
        Executes a Quantum Split Move. A piece moving from a deterministic state splits 
        into two target squares simultaneously, creating an equal superposition (50% / 50%).
        """
        if piece_name not in self.piece_positions or from_coord not in self.piece_positions[piece_name]:
            raise ValueError(f"Piece {piece_name} does not occupy the specified source coordinate.")
        
        fx, fy = from_coord
        tax, tay = to_coord_a
        tbx, tby = to_coord_b

        # Clear the original square
        self.grid[fx, fy] = None
        
        # Create equal probability amplitudes for both target coordinates: 1/sqrt(2)
        amplitude = 1.0 / np.sqrt(2)
        self.quantum_state.apply_superposition(tax, tay, amplitude + 0j)
        self.quantum_state.apply_superposition(tbx, tby, amplitude + 0j)
        
        # Update registries
        self.grid[tax, tay] = piece_name
        self.grid[tbx, tby] = piece_name
        self.piece_positions[piece_name] = [to_coord_a, to_coord_b]
        
        # Entangle the two target squares as they share the same physical piece state
        self.entanglement_manager.create_entanglement(to_coord_a, to_coord_b)

    def execute_merge_move(self, piece_name: str, from_coord_a: Tuple[int, int], 
                           from_coord_b: Tuple[int, int], to_coord: Tuple[int, int]) -> None:
        """
        Executes a Quantum Merge Move. Recombines a piece existing in a superposition of 
        two squares back into a single target square deterministically.
        """
        positions = self.piece_positions.get(piece_name, [])
        if from_coord_a not in positions or from_coord_b not in positions:
            raise ValueError(f"Piece {piece_name} is not distributed across both source squares.")

        # Clear both source locations
        self.grid[from_coord_a[0], from_coord_a[1]] = None
        self.grid[from_coord_b[0], from_coord_b[1]] = None
        
        # Collapse the superposition mathematically back into a deterministic 100% state
        tx, ty = to_coord
        self._set_piece_deterministically(piece_name, tx, ty)
        
        # Clear previous entanglement bonds since it has merged completely
        self.entanglement_manager.trigger_collapse_chain(from_coord_a)
        self.entanglement_manager.trigger_collapse_chain(from_coord_b)
