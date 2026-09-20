import numpy as np
from typing import List, Tuple
from src.qml.models import HybridQuantumValueNetwork

class QuantumReinforcementAgent:
    """
    Implements a Quantum-Enhanced Reinforcement Learning Agent using policy execution,
    reward tracking, and exploration-exploitation decay mechanics.
    """
    def __init__(self, num_actions: int = 4, epsilon_init: float = 1.0, epsilon_decay: float = 0.995):
        self.num_actions = num_actions
        # Quantum network acting as our Q-value approximation engine
        self.quantum_network = HybridQuantumValueNetwork(num_qubits=num_actions)
        
        # Hyperparameters for Deep Q-Learning style mechanics
        self.epsilon = epsilon_init
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.gamma = 0.95  # Discount factor for future quantum rewards

    def select_action(self, state_features: List[float], legal_moves: List[int]) -> int:
        """
        Selects an action using the Epsilon-Greedy policy. Balances exploring random moves 
        with exploiting known optimal states calculated by the hybrid quantum circuit.
        """
        if not legal_moves:
            raise ValueError("The list of legal actions cannot be empty.")

        # Exploration phase: Select a random valid action
        if np.random.rand() <= self.epsilon:
            return int(np.random.choice(legal_moves))
        
        # Exploitation phase: Query the parametric quantum circuit for state-action values
        q_values = self.quantum_network.forward_pass(state_features)
        
        # Filter action outputs to extract the maximum value among valid legal moves only
        best_action = legal_moves[0]
        max_q = float('-inf')
        
        for move in legal_moves:
            if 0 <= move < self.num_actions:
                if q_values[move] > max_q:
                    max_q = q_values[move]
                    best_action = move
                    
        return int(best_action)

    def learn_from_transition(self, state: List[float], action: int, reward: float, 
                              next_state: List[float], legal_next_moves: List[int], done: bool) -> float:
        """
        Updates the parametric quantum weights based on Temporal Difference (TD) target calculations.
        """
        current_predictions = self.quantum_network.forward_pass(state)
        target_q_values = np.array(current_predictions.copy())

        if done:
            target_q_values[action] = reward
        else:
            # Calculate future reward expectations from the next quantum state vector
            next_q_values = self.quantum_network.forward_pass(next_state)
            max_future_q = max([next_q_values[m] for m in legal_next_moves]) if legal_next_moves else 0.0
            
            # Bellman equation update mapping
            target_q_values[action] = reward + self.gamma * max_future_q

        # Optimize the hybrid circuit parameters using parameter-shift gradients
        loss = self.quantum_network.train_step(state, list(target_q_values))
        
        # Decay exploration rate safely to increase exploitation confidence over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
