import numpy as np
from typing import Dict, List, Tuple

class EntanglementManager:
    """
    Manages quantum entanglement linkages between distinct board coordinates.
    Tracks collapse dependencies to ensure that observing one entangled entity
    instantaneously updates or collapses its coupled counterpart.
    """
    def __init__(self):
        # Maps a specific coordinate tuple (x, y) to its entangled partner coordinates
        self.entanglement_registry: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def create_entanglement(self, coord_a: Tuple[int, int], coord_b: Tuple[int, int]) -> None:
        """
        Creates a bidirectional quantum entanglement link between coordinate A and coordinate B.
        """
        if coord_a == coord_b:
            raise ValueError("An element cannot be entangled with its own exact coordinate.")

        if coord_a not in self.entanglement_registry:
            self.entanglement_registry[coord_a] = []
        if coord_b not in self.entanglement_registry:
            self.entanglement_registry[coord_b] = []

        if coord_b not in self.entanglement_registry[coord_a]:
            self.entanglement_registry[coord_a].append(coord_b)
        if coord_a not in self.entanglement_registry[coord_b]:
            self.entanglement_registry[coord_b].append(coord_a)

    def get_entangled_partners(self, coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Retrieves all coordinates that are quantum-entangled with the given coordinate.
        """
        return self.entanglement_registry.get(coord, [])

    def trigger_collapse_chain(self, collapsed_coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Triggers a cascading wave function collapse. When an entangled coordinate collapses,
        all registered partners are identified so their states can be updated simultaneously.
        Clears the entanglement link post-collapse as the system becomes deterministic.
        """
        if collapsed_coord not in self.entanglement_registry:
            return []

        affected_partners = self.entanglement_registry[collapsed_coord].copy()

        # Remove the entanglement links since measurement has occurred (decoherence)
        for partner in affected_partners:
            if partner in self.entanglement_registry and collapsed_coord in self.entanglement_registry[partner]:
                self.entanglement_registry[partner].remove(collapsed_coord)
                if not self.entanglement_registry[partner]:
                    del self.entanglement_registry[partner]

        del self.entanglement_registry[collapsed_coord]
        return affected_partners
