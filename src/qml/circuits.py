import pennylane as qml
from pennylane import numpy as pnp
from typing import List

class ParameterizedQuantumCircuit:
    """
    Defines trainable Parametric Quantum Circuits (PQC) using PennyLane to encode 
    probabilistic game board states into quantum registers for AI evaluation.
    """
    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits
        # Initialize a high-performance lightning simulator device provided by PennyLane
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.circuit = self._create_quantum_node()

    def _create_quantum_node(self):
        """
        Internal helper to construct a quantum node (QNode) that binds the simulator 
        device to the mathematical circuit execution pipeline.
        """
        @qml.qnode(self.dev)
        def quantum_circuit(features: pnp.ndarray, weights: pnp.ndarray):
            """
            Executes amplitude embedding for feature mapping and applies layers of 
            parameterized rotations coupled with strongly entangling gates.
            """
            # Step 1: Quantum Feature Map (Encode game board probabilities into qubits)
            # Normalizes features internally to map safely onto the unit sphere
            qml.AmplitudeEmbedding(features, wires=range(self.num_qubits), normalize=True)

            # Step 2: Trainable Ansatz (Parameterized Quantum Neural Network layers)
            # Iterates over weight tensor layers to apply rotational shifting
            layers = weights.shape[0]
            for layer in range(layers):
                for qubit in range(self.num_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)

                # Linear entanglement chain (CNOT gates) to lock state dependencies
                for qubit in range(self.num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                if self.num_qubits > 2:
                    qml.CNOT(wires=[self.num_qubits - 1, 0])

            # Step 3: Measurement (Expectation value of PauliZ operator acting as prediction space)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.num_qubits)]

        return quantum_circuit

    def generate_initial_weights(self, layers: int = 2) -> pnp.ndarray:
        """
        Generates uniform random initial weight matrices matching the ansatz layer structure.
        Shape format: (layers, num_qubits, 3 rotation paths [X, Y, Z])
        """
        return pnp.random.uniform(low=-pnp.pi, high=pnp.pi, size=(layers, self.num_qubits, 3), requires_grad=True)
