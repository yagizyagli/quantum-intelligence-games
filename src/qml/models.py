import pennylane as qml
from pennylane import numpy as pnp
from typing import List, Tuple
from src.qml.circuits import ParameterizedQuantumCircuit

class HybridQuantumValueNetwork:
    """
    Implements a Hybrid Classical-Quantum Neural Network (HQNN) designed to evaluate 
    probabilistic board game states and predict optimum move values.
    """
    def __init__(self, num_qubits: int = 4, num_layers: int = 2, learning_rate: float = 0.05):
        self.pqc_module = ParameterizedQuantumCircuit(num_qubits=num_qubits)
        self.learning_rate = learning_rate
        
        # Initialize trainable parameters for the quantum circuit layers
        self.weights = self.pqc_module.generate_initial_weights(layers=num_layers)
        
        # Classical bias vector to complement quantum expectation outputs
        self.bias = pnp.array([0.0] * num_qubits, requires_grad=True)

    def forward_pass(self, board_state_features: List[float]) -> pnp.ndarray:
        """
        Executes a hybrid forward pass. Passes classical state probabilities through 
        the quantum register and adjusts the expectation values with a classical bias.
        """
        # Convert raw Python list features into a differentiable quantum array
        features = pnp.array(board_state_features, requires_grad=False)
        
        # Execute the parametric quantum circuit (Expectation values between -1.0 and 1.0)
        quantum_outputs = self.pqc_module.circuit(features, self.weights)
        quantum_outputs = pnp.array(quantum_outputs)
        
        # Combine quantum feature processing with classical bias shift
        hybrid_predictions = quantum_outputs + self.bias
        return hybrid_predictions

    def compute_loss(self, predictions: pnp.ndarray, target_values: pnp.ndarray) -> pnp.ndarray:
        """
        Calculates the Mean Squared Error (MSE) loss function for value evaluation tracking.
        """
        return pnp.mean((predictions - target_values) ** 2)

    def train_step(self, board_state_features: List[float], target_values: List[float]) -> float:
        """
        Performs a single hybrid optimization step. Computes analytical quantum gradients 
        via Parameter-Shift rules and updates both weights and classical biases.
        """
        targets = pnp.array(target_values, requires_grad=False)

        # Define a dynamic objective function local to the training step for the gradient optimizer
        def cost_function(trainable_params: Tuple[pnp.ndarray, pnp.ndarray]) -> pnp.ndarray:
            w, b = trainable_params
            # Temporarily override weights and biases to trace execution gradients
            self.weights = w
            self.bias = b
            preds = self.forward_pass(board_state_features)
            return self.compute_loss(preds, targets)

        # Instantiate PennyLane's specialized autograd Nesterov Momentum Optimizer
        optimizer = qml.NesterovMomentumOptimizer(stepsize=self.learning_rate)
        
        # Trigger gradient calculation and backward propagation pass
        (updated_weights, updated_bias), current_loss = optimizer.step_and_cost(
            cost_function, (self.weights, self.bias)
        )
        
        # Commit updated parameters back to the live model register
        self.weights = updated_weights
        self.bias = updated_bias
        
        return float(current_loss)
