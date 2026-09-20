import numpy as np
from typing import List, Union

class QuantumTransformationHelpers:
    """
    Provides high-performance mathematical utility functions for tensor normalization,
    vector scaling, and quantum feature mapping conversions.
    """
    
    @staticmethod
    def normalize_to_quantum_amplitudes(data_vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Converts a standard classical feature array into an L2-normalized vector.
        This ensures the sum of absolute squared values equals 1.0 (|psi|^2 = 1),
        making it mathematically compatible with Quantum Amplitude Embedding.
        """
        vector = np.array(data_vector, dtype=float)
        l2_norm = np.linalg.norm(vector)
        
        if l2_norm == 0:
            # Prevent division by zero; return a uniform superposition probability array
            length = len(vector)
            return np.ones(length, dtype=float) / np.sqrt(length)
            
        return vector / l2_norm

    @staticmethod
    def matrix_to_flat_probabilities(matrix: np.ndarray) -> np.ndarray:
        """
        Takes a multi-dimensional state matrix, computes its raw probability distribution,
        and flattens it safely into a 1D probability density array.
        """
        probabilities = np.abs(matrix) ** 2
        total_sum = np.sum(probabilities)
        
        if total_sum > 0:
            probabilities /= total_sum
        else:
            # If completely empty, return uniform distribution across the flattened shape
            probabilities = np.ones(matrix.shape, dtype=float) / matrix.size
            
        return probabilities.flatten()

    @staticmethod
    def scale_quantum_expectations(expectation_values: np.ndarray, target_min: float = 0.0, 
                                   target_max: float = 1.0) -> np.ndarray:
        """
        Scales quantum circuit expectation values (which naturally fall between -1.0 and 1.0)
        into a targeted classical range, such as [0.0, 1.0] for standard probability evaluation.
        """
        # Min-max scaling formula mapped from [-1, 1] input range
        scaled_values = (expectation_values + 1.0) / 2.0
        # Re-scale to fit custom user bounds dynamically
        scaled_values = scaled_values * (target_max - target_min) + target_min
        return scaled_values
