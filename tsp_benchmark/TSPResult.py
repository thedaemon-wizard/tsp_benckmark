from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

class TSPResult:
    """Class for storing TSP solution with enhanced evaluation features"""
    def __init__(self, path: List[int], distance: float,
                 feasible: bool, algorithm: str,
                 computation_time: float = None,
                 additional_info: Dict = None):
        self.path = path
        self.distance = distance
        self.feasible = feasible
        self.algorithm = algorithm
        self.computation_time = computation_time
        self.additional_info = additional_info or {}

        # Add evaluation metrics
        self.evaluation_metrics = {}

    def __repr__(self):
        return (f"TSPResult(algorithm={self.algorithm}, "
                f"distance={self.distance:.5f}, "
                f"feasible={self.feasible}, "
                f"computation_time={self.computation_time:.5f}s)")

    def to_dict(self) -> Dict:
        """Convert result to dictionary format (JSON serializable)"""
        # Convert additional_info to JSON serializable format
        serializable_info = {}
        for key, value in self.additional_info.items():
            if key == 'sampleset':
                # Exclude SampleSet object
                continue
            elif key == 'circuit':
                # Exclude circuit objects (not JSON serializable)
                # Instead, add circuit metadata if available
                if isinstance(value, dict):
                    circuit_metadata = {}
                    if 'original' in value and hasattr(value['original'], 'num_qubits'):
                        circuit_metadata['original_qubits'] = value['original'].num_qubits
                        circuit_metadata['original_depth'] = value['original'].depth()
                        circuit_metadata['original_gates'] = value['original'].size()
                    if 'transpiled' in value and hasattr(value['transpiled'], 'num_qubits'):
                        circuit_metadata['transpiled_depth'] = value['transpiled'].depth()
                        circuit_metadata['transpiled_gates'] = value['transpiled'].size()
                    if circuit_metadata:
                        serializable_info['circuit_metadata'] = circuit_metadata
                continue
            elif key == 'optimal_params' and hasattr(value, 'tolist'):
                # Convert numpy array to list
                serializable_info[key] = value.tolist()
            elif isinstance(value, (np.ndarray, np.generic)):
                # Convert other numpy types to list
                serializable_info[key] = value.tolist()
            elif isinstance(value, (bool, int, str, list, dict, type(None))):
                # Keep basic types as is
                serializable_info[key] = value
            elif isinstance(value, float):
                # Handle float values including inf
                if value == float('inf'):
                    serializable_info[key] = 'inf'
                elif value == float('-inf'):
                    serializable_info[key] = '-inf'
                elif np.isnan(value):
                    serializable_info[key] = 'nan'
                else:
                    serializable_info[key] = value
            else:
                # Convert other objects to string representation
                serializable_info[key] = str(value)

        # Handle distance value
        if self.distance == float('inf'):
            distance_value = 'inf'
        elif self.distance == float('-inf'):
            distance_value = '-inf'
        elif isinstance(self.distance, float) and np.isnan(self.distance):
            distance_value = 'nan'
        else:
            distance_value = float(self.distance)

        return {
            'algorithm': self.algorithm,
            'path': self.path,
            'distance': distance_value,
            'feasible': self.feasible,
            'computation_time': self.computation_time,
            'evaluation_metrics': {
                k: (float(v) if isinstance(v, (int, float)) and v != float('inf') else
                    ('inf' if v == float('inf') else v))
                for k, v in self.evaluation_metrics.items()
            },
            'additional_info': serializable_info
        }