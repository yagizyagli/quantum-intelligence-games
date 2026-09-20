import unittest
import numpy as np
from src.core.state import QuantumState
from src.core.entanglement import EntanglementManager

class TestQuantumCoreEngine(unittest.TestCase):
    """
    Automated unit testing suite validating the mathematical accuracy of 
    the core quantum state simulator and entanglement cascade managers.
    """

    def setUp(self):
        """Initializes testing fixtures prior to executing each distinct test case."""
        self.board_size = 8
        self.quantum_state = QuantumState(size=self.board_size)
        self.entanglement_manager = EntanglementManager()

    def test_superposition_and_probability_normalization(self):
        """Validates that applying complex amplitudes correctly normalizes total probability to 1.0."""
        # Setup an equal superposition across two specific tiles: 1/sqrt(2) state
        amplitude = 1.0 / np.sqrt(2)
        self.quantum_state.apply_superposition(0, 0, amplitude + 0j)
        self.quantum_state.apply_superposition(0, 1, amplitude + 0j)

        probabilities = self.quantum_state.calculate_probabilities()
        
        # Total probability of the closed system must mathematically equal 1.0
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=5)
        # Each superposition path must hold exactly 50% probability
        self.assertAlmostEqual(float(probabilities[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(probabilities[0, 1]), 0.5, places=5)

    def test_wave_function_collapse(self):
        """Validates that measurement forces a total collapse of the wave function into a deterministic state."""
        amplitude = 1.0 / np.sqrt(2)
        self.quantum_state.apply_superposition(2, 2, amplitude + 0j)
        self.quantum_state.apply_superposition(2, 3, amplitude + 0j)

        # Trigger measurement observation event
        cx, cy = self.quantum_state.measure_and_collapse()

        # The collapsed coordinate must belong to one of the initial superposition paths
        self.assertIn((cx, cy), [(2, 2), (2, 3)])

        # Post-collapse probability of the chosen tile must equal exactly 100% (1.0)
        post_probabilities = self.quantum_state.calculate_probabilities()
        self.assertEqual(float(post_probabilities[cx, cy]), 1.0)

    def test_entanglement_linkages_and_cascade_collapse(self):
        """Validates bidirectional entanglement mapping and the subsequent cascading decoherence chain."""
        coord_a = (4, 4)
        coord_b = (4, 5)

        # Create quantum link configuration
        self.entanglement_manager.create_entanglement(coord_a, coord_b)
        
        # Verify bidirectional structural registration
        partners_of_a = self.entanglement_manager.get_entangled_partners(coord_a)
        self.assertIn(coord_b, partners_of_a)

        # Trigger observation event on coordinate A to test cascade collapse
        affected_partners = self.entanglement_manager.trigger_collapse_chain(coord_a)
        
        # Coordinate B must be immediately swept and signaled in the affected partner list
        self.assertIn(coord_b, affected_partners)
        
        # Post-collapse check: The entanglement registry must be cleared (decoherence)
        self.assertEqual(len(self.entanglement_manager.get_entangled_partners(coord_a)), 0)
        self.assertEqual(len(self.entanglement_manager.get_entangled_partners(coord_b)), 0)

if __name__ == "__main__":
    unittest.main()
