import unittest
from pennylane import numpy as pnp
from src.qml.circuits import ParameterizedQuantumCircuit
from src.qml.models import HybridQuantumValueNetwork
from src.qml.reinforcement import QuantumReinforcementAgent

class TestQuantumMachineLearningComponents(unittest.TestCase):
    """
    Automated unit testing suite validating PennyLane parametric circuit gradients,
    hybrid forward loops, and reinforcement learning selection decay.
    """

    def setUp(self):
        """Initializes testing fixtures before each distinct test case."""
        self.num_actions = 4
        self.agent = QuantumReinforcementAgent(num_actions=self.num_actions)

    def test_hybrid_network_forward_pass(self):
        """Validates that classical inputs successfully map onto expectation scalar arrays."""
        # Generate dummy normalized state vector inputs (4 metrics for 4 qubits)
        dummy_state = [0.5, 0.5, 0.5, 0.5]
        predictions = self.agent.quantum_network.forward_pass(dummy_state)
        
        # Output shape length must match the mapped action register count
        self.assertEqual(len(predictions), self.num_actions)
        self.assertIsInstance(predictions, pnp.tensor)

    def test_epsilon_greedy_decay(self):
        """Validates that reinforcement training updates decay the exploration factor correctly."""
        initial_epsilon = self.agent.epsilon
        
        dummy_state = [0.5, 0.5, 0.5, 0.5]
        # Simulate a transition learn update
        self.agent.learn_from_transition(
            state=dummy_state, action=0, reward=1.0, 
            next_state=dummy_state, legal_next_moves=[0, 1], done=False
        )
        
        # Epsilon must decrease to push the policy toward optimal exploitation mapping
        self.assertLess(self.agent.epsilon, initial_epsilon)

if __name__ == "__main__":
    unittest.main()
