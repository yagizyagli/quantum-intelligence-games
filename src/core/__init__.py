"""
Core Quantum Simulation Layer.
Exposes high-performance classes for managing superpositions, state vector collapses,
and cascading quantum entanglement linkages across modular board games.
"""

from src.core.state import QuantumState
from src.core.entanglement import EntanglementManager

__all__ = [
    "QuantumState",
    "EntanglementManager",
]
