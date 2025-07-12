import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import time
from itertools import combinations
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import json
import csv
import os
from pathlib import Path

# sklearn for MDS (Multi-Dimensional Scaling)
try:
    from sklearn.manifold import MDS
    MDS_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not available. MDS-based coordinate estimation will not work.")
    MDS_AVAILABLE = False

# OMMX related
from ommx.v1 import DecisionVariable, Instance
import ommx_openjij_adapter as oj_ad
import openjij as oj

# Qiskit related (for QAOA) - Using V2 primitives
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# AerSimulator V2 primitives setup
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit import transpile

# Use scipy's minimize directly (Qiskit v2.0 recommended)
from scipy.optimize import minimize


from .Algorithm import *
from .TSPResult import *
        

class TSPBenchmark:
    """Benchmark library for Traveling Salesman Problem (Qiskit v2.0 compatible)"""

    def __init__(self, distance_matrix: np.ndarray = None, coordinates: List[Tuple[float, float]] = None):
        """
        Parameters
        ----------
        distance_matrix : np.ndarray, optional
            Distance matrix where d[i][j] is the distance from city i to city j
        coordinates : List[Tuple[float, float]], optional
            List of city coordinates. If provided, distance matrix will be generated automatically
        """
        if coordinates is not None:
            self.coordinates = coordinates
            self.distance_matrix = self.create_distance_matrix_from_coordinates(coordinates)
            self.coordinates_estimated = False
        elif distance_matrix is not None:
            self.distance_matrix = np.array(distance_matrix)
            # Estimate coordinates from distance matrix
            self.coordinates = self._estimate_coordinates_from_distance_matrix(self.distance_matrix)
            self.coordinates_estimated = True
        else:
            raise ValueError("Either distance_matrix or coordinates must be provided")

        self.n_cities = len(self.distance_matrix)
        self.results_history = []  # Store results history

        # Validate distance matrix
        self._validate_distance_matrix()

    def _estimate_coordinates_from_distance_matrix(self, distance_matrix: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Estimate coordinates from distance matrix using MDS"""
        if not MDS_AVAILABLE:
            print("Warning: sklearn not available. Coordinates will not be estimated.")
            return None

        try:
            # Use MDS (Multi-Dimensional Scaling) to estimate 2D coordinates
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords = mds.fit_transform(distance_matrix)

            # Convert to list of tuples
            coordinates = [(float(coord[0]), float(coord[1])) for coord in coords]
            print(f"Coordinates estimated from distance matrix using MDS (stress: {mds.stress_:.5f})")
            return coordinates
        except Exception as e:
            print(f"Failed to estimate coordinates from distance matrix: {e}")
            return None

    @staticmethod
    def create_distance_matrix_from_coordinates(coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """Generate distance matrix from coordinate list"""
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = coordinates[i]
                    x2, y2 = coordinates[j]
                    distance_matrix[i][j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        return distance_matrix

    def _validate_distance_matrix(self):
        """Validate distance matrix"""
        if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")

        if np.any(np.diag(self.distance_matrix) != 0):
            print("Warning: Diagonal elements of distance matrix should be 0")

    def solve(self, algorithm: Union[Algorithm, str], **kwargs) -> TSPResult:
        """
        Solve TSP using specified algorithm

        Parameters
        ----------
        algorithm : Algorithm or str
            Algorithm to use
        **kwargs : dict
            Algorithm-specific parameters

        Returns
        -------
        TSPResult
            Object containing solution information
        """
        if isinstance(algorithm, str):
            algorithm = Algorithm(algorithm)

        if algorithm == Algorithm.SIMULATED_ANNEALING:
            result = self._solve_simulated_annealing(**kwargs)
        elif algorithm == Algorithm.QAOA:
            result = self._solve_qaoa(**kwargs)
        elif algorithm == Algorithm.HELD_KARP:
            result = self._solve_held_karp(**kwargs)
        else:
            raise NotImplementedError(f"Algorithm {algorithm} is not implemented yet")

        # Evaluate solution
        self._evaluate_solution(result)

        # Save result to history
        self.results_history.append(result)

        return result

    def _create_ommx_instance(self) -> Instance:
        """Create OMMX Instance"""
        N = self.n_cities

        # Create decision variables
        x = [[
            DecisionVariable.binary(
                i + N * t,
                name="x",
                subscripts=[t, i]
            )
            for i in range(N)
        ] for t in range(N)]

        # Objective function: minimize total travel distance
        objective = sum(
            self.distance_matrix[i][j] * x[t][i] * x[(t+1) % N][j]
            for i in range(N)
            for j in range(N)
            for t in range(N)
        )

        # Constraint 1: At each time, salesman is at exactly one city
        time_constraints = [
            (sum(x[t][i] for i in range(N)) == 1)
            .set_id(t)
            .add_name("onehot_time")
            .add_subscripts([t])
            for t in range(N)
        ]

        # Constraint 2: Each city is visited exactly once
        location_constraints = [
            (sum(x[t][i] for t in range(N)) == 1)
            .set_id(i + N)
            .add_name("onehot_location")
            .add_subscripts([i])
            for i in range(N)
        ]

        # Create instance
        instance = Instance.from_components(
            decision_variables=[x[t][i] for t in range(N) for i in range(N)],
            objective=objective,
            constraints=time_constraints + location_constraints,
            sense=Instance.MINIMIZE
        )

        return instance

    def _solve_simulated_annealing(self,
                                   num_reads: int = 100,
                                   uniform_penalty_weight: float = 20.0,
                                   **sa_params) -> TSPResult:
        """Solve problem using Simulated Annealing (with OMMX)"""
        start_time = time.time()

        # Create OMMX Instance
        instance = self._create_ommx_instance()

        # Sample using OMMX-OpenJij-Adapter
        adapter = oj_ad.OMMXOpenJijSAAdapter(instance)
        sampleset = adapter.sample(
            instance,
            num_reads=num_reads,
            uniform_penalty_weight=uniform_penalty_weight,
            **sa_params
        )

        # Get best feasible solution
        best_sample = sampleset.best_feasible_unrelaxed()
        if best_sample is None:
            best_sample = sampleset.best()
            print("Warning: No feasible solution found. Returning best solution.")
            feasible = False
        else:
            feasible = True

        # Construct tour from solution
        x_value = best_sample.extract_decision_variables("x")
        path = self._extract_path(x_value)

        computation_time = time.time() - start_time

        return TSPResult(
            path=path,
            distance=best_sample.objective,
            feasible=feasible,
            algorithm="simulated_annealing",
            computation_time=computation_time,
            additional_info={
                'num_reads': num_reads,
                'penalty_weight': uniform_penalty_weight,
                'sampleset': sampleset,
                'num_feasible_solutions': len(sampleset.summary.query("feasible == True"))
            }
        )

    def _solve_qaoa(self,
                    p: int = 2,
                    optimizer: str = 'COBYLA',
                    maxiter: int = 100,
                    uniform_penalty_weight: float = 20.0,
                    shots: int = 2048,
                    backend_type: str = 'auto',
                    use_gpu: bool = False,
                    **kwargs) -> TSPResult:
        """Solve problem using QAOA (Qiskit v2.0 compatible)"""
        start_time = time.time()

        try:
            # Get QUBO matrix from OMMX Instance
            instance = self._create_ommx_instance()
            qubo_matrix, constant = instance.to_qubo(uniform_penalty_weight=uniform_penalty_weight)

            # Convert QUBO to Ising Hamiltonian
            n_vars = self.n_cities * self.n_cities

            # Warning for large number of qubits
            if n_vars > 20:
                print(f"  Warning: {n_vars} qubits may require significant computation time.")
                print(f"  Consider reducing the number of cities.")

            J, h, offset = self._qubo_to_ising(qubo_matrix, n_vars)

            # Create Pauli Hamiltonian
            cost_hamiltonian = self._ising_to_pauli_hamiltonian(J, h)

            # Check problem scale
            max_coeff = max(abs(coeff) for _, coeff in cost_hamiltonian.to_list())
            print(f"  Max coefficient: {max_coeff:.5f}")

            # Scale if necessary
            if max_coeff > 100:
                scaling_factor = 10.0 / max_coeff
                cost_hamiltonian = cost_hamiltonian * scaling_factor
                print(f"  Scaling Hamiltonian: {scaling_factor:.5f}")

            # Create QAOA ansatz
            ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
            ansatz.measure_all()
            
            #Store circuit separately, not in additional_info that gets serialized
            self._last_qaoa_circuit = {
                'original': ansatz.copy(),
                'transpiled': None,
                'parametrized': True,
                'optimization_level': 2
            }

            # Auto-select backend
            aer_backend = self._select_optimal_backend(backend_type, use_gpu, n_vars)

            # Optimize transpilation
            transpiled_ansatz = transpile(ansatz, backend=aer_backend, optimization_level=2)

            #pm = generate_preset_pass_manager(backend=aer_backend, optimization_level=2)
            #transpiled_ansatz = pm.run(ansatz)

            print(f"  Number of qubits: {n_vars}, QAOA layers: {p}")
            print(f"  Backend: {aer_backend}")
            print(f"  Circuit parameters: {len(transpiled_ansatz.parameters)}")

            # Set initial parameters
            np.random.seed(42)
            # Trotterized Quantum Annealing (TQA) inspired initialization
            init_params = []

            # β (mixer) parameters - decreasing schedule
            for i in range(p):
                beta_val = np.pi * (1 - (i + 1) / (p + 1))
                init_params.append(beta_val)

            # γ (cost) parameters - increasing schedule
            for i in range(p):
                gamma_val = np.pi * (i + 1) / (2 * p)
                init_params.append(gamma_val)

            init_params = np.array(init_params)

            print(f"  Initial parameters: β={init_params[:p]}, γ={init_params[p:]}")

            # Use EstimatorV2
            estimator = EstimatorV2().from_backend(backend=aer_backend)

            # Cost function history
            cost_history = []
            parameter_history = []
            iteration_count = 0
            best_cost = float('inf')
            best_params = None

            def cost_func_estimator(params):
              nonlocal iteration_count, best_cost, best_params
              iteration_count += 1

              try:
                  # Limit parameter range
                  params = np.array(params)
                  params[:p] = np.clip(params[:p], 0, np.pi)  # β ∈ [0, π]
                  params[p:] = np.clip(params[p:], 0, 2*np.pi)  # γ ∈ [0, 2π]

                  # EstimatorV2 PUB format
                  pub = (transpiled_ansatz, [cost_hamiltonian], params)
                  job = estimator.run([pub])
                  result = job.result()

                  # Get cost value
                  cost = float(result[0].data.evs[0])

                  # Reverse scaling
                  if max_coeff > 100:
                      cost = cost / scaling_factor

                  cost = cost + offset + constant

                  cost_history.append(cost)
                  parameter_history.append(params.copy())

                  if cost < best_cost:
                      best_cost = cost
                      best_params = params.copy()

                  if iteration_count % 10 == 0:
                      print(f"  Iteration {iteration_count}: cost = {cost:.4f}")

                  return cost

              except Exception as e:
                  print(f"Error in cost function: {e}")
                  return 1e10


            if optimizer == 'COBYLA':
                options = {
                    'maxiter': maxiter,
                    'rhobeg': 0.5,  # Initial step size
                    'tol': 1e-6
                }
            else:
                options = {'maxiter': maxiter}

            # Run optimization
            result = minimize(
                cost_func_estimator,
                init_params,
                method=optimizer,
                options=options
            )



            print(f"QAOA optimization completed in {iteration_count} iterations")
            print(f"Final cost: {result.fun:.4f}")
            print(f"Optimization success: {getattr(result, 'success', True)}")

            # Use best parameters
            if best_params is not None:
                optimal_params = best_params
                print(f"  Best cost: {best_cost:.4f}")
            else:
                optimal_params = result.x
            
            # In Qiskit v2.0, use assign_parameters instead of bind_parameters
            try:
                # Create bound circuits using assign_parameters (Qiskit v2.0 compatible)
                bound_ansatz = ansatz.assign_parameters(optimal_params)
                bound_transpiled = transpiled_ansatz.assign_parameters(optimal_params)
                
                # Update stored circuits with bound versions
                self._last_qaoa_circuit['bound_original'] = bound_ansatz
                self._last_qaoa_circuit['bound_transpiled'] = bound_transpiled
                self._last_qaoa_circuit['optimal_params'] = optimal_params
                
                print(f"  Successfully created bound circuits with optimal parameters")
            except Exception as e:
                print(f"  Warning: Could not create bound circuits: {e}")
                # Fallback: store optimal parameters separately
                self._last_qaoa_circuit['optimal_params'] = optimal_params


            # Use SamplerV2
            sampler = SamplerV2().from_backend(backend=aer_backend)

            # Sampling
            pub = (transpiled_ansatz, optimal_params)
            job = sampler.run([pub], shots=shots)
            result_sampler = job.result()

            # Get measurement results
            pub_result = result_sampler[0]
            counts = {}
            if hasattr(pub_result.data, 'meas'):
                counts = pub_result.data.meas.get_counts()

            # Select the most probable bitstring
            if not counts:
                print("Warning: No measurement results obtained")
                return TSPResult(
                    path=[],
                    distance=float('inf'),
                    feasible=False,
                    algorithm="qaoa",
                    computation_time=time.time() - start_time,
                    additional_info={'error': 'No measurement results'}
                )

            # Evaluate top bitstrings
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

            best_feasible_path = None
            best_feasible_distance = float('inf')

            # Evaluate top 10 solutions
            for bitstring, count in sorted_counts[:10]:
                if len(bitstring) < n_vars:
                    bitstring = bitstring.zfill(n_vars)

                x_values = [int(bit) for bit in bitstring[::-1]]
                path = self._extract_path_from_binary(x_values)

                # Check feasibility
                if self._check_feasibility_x(x_values) and len(path) == self.n_cities:
                    distance = self.evaluate_total_distance(path)
                    if distance < best_feasible_distance:
                        best_feasible_distance = distance
                        best_feasible_path = path
                        print(f"  Feasible solution found: distance={distance:.5f}, count={count}")

            # If no feasible solution found, relax constraints
            if best_feasible_path is None:
                print("  No feasible solution found. Using most frequent solution.")
                best_bitstring = sorted_counts[0][0]
                if len(best_bitstring) < n_vars:
                    best_bitstring = best_bitstring.zfill(n_vars)
                x_values = [int(bit) for bit in best_bitstring[::-1]]
                best_feasible_path = self._extract_path_from_binary(x_values)
                best_feasible_distance = self.evaluate_total_distance(best_feasible_path) if best_feasible_path else float('inf')
                feasible = False
            else:
                feasible = True

            computation_time = time.time() - start_time

            return TSPResult(
                path=best_feasible_path if best_feasible_path else [],
                distance=best_feasible_distance,
                feasible=feasible,
                algorithm="qaoa",
                computation_time=computation_time,
                additional_info={
                    'p': p,
                    'optimizer': optimizer,
                    'optimal_params': optimal_params.tolist() if hasattr(optimal_params, 'tolist') else optimal_params,
                    'cost_history': cost_history,
                    'final_cost': float(min(cost_history)) if cost_history else float('inf'),
                    'counts': dict(sorted_counts[:10]),
                    'iterations': iteration_count,
                    'shots': shots,
                    'penalty_weight': uniform_penalty_weight,
                    'max_hamiltonian_coeff': float(max_coeff),
                    'optimization_success': getattr(result, 'success', True),
                    'backend_type': backend_type,
                    'use_gpu': use_gpu,
                    'backend_name': str(aer_backend)
                }
            )

        except Exception as e:
            print(f"Error in QAOA execution: {e}")
            import traceback
            traceback.print_exc()
            return TSPResult(
                path=[],
                distance=float('inf'),
                feasible=False,
                algorithm="qaoa",
                computation_time=time.time() - start_time,
                additional_info={'error': str(e)}
            )



    def _select_optimal_backend(self, backend_type: str, use_gpu: bool, n_vars: int) -> AerSimulator:
        """Selecting the optimal backend (Quantum Bit Limit Avoidance version)"""
        
        # Check GPU availability
        try:
            temp_simulator = AerSimulator()
            available_devices = temp_simulator.available_devices()
            gpu_available = 'GPU' in available_devices
        except Exception as e:
            print(f"  Warning: Error in GPU availability check: {e}")
            gpu_available = False
            available_devices = ['CPU']
        
        if backend_type == 'auto':
            if use_gpu and gpu_available:
                # Setting to avoid qubit limit when using GPU
                if n_vars <= 30:
                    method = 'statevector'
                #elif n_vars <= 30:
                #    method = 'density_matrix'
                else:
                    method = 'tensor_network'  # For large circuits
                
                device = 'GPU'
                print(f"  Auto Selection: GPU Use (method={method})")
            else:
                # CPU使用時
                if n_vars <= 25:
                    method = 'statevector'
                else:
                    method = 'automatic'
                device = 'CPU'
                print(f"   Auto Selection: CPU Use (method={method})")
        else:
            method = backend_type
            device = 'GPU' if (use_gpu and gpu_available) else 'CPU'
        
        try:
            # Setup for avoiding qubit limit
            backend_options = {
                'method': method,
                'device': device,
            }
            
            # Limit Avoidance Settings for GPU Use
            if device == 'GPU':
                
                # Additional settings for large circuits
                if n_vars > 30:
                    # Distributed processing with blocking function
                    blocking_qubits = min(25, max(20, n_vars - 5))
                    backend_options.update({
                         #'precision': 'single',  # Reduced memory usage
                        'max_memory_mb': -1,  # Disable memory limit
                        #'max_parallel_threads': 1,
                        #'max_parallel_experiments': 1,
                        'enable_truncation': True,  # Automatic deletion of unnecessary qubits
                        'blocking_enable': True,
                        'blocking_qubits': blocking_qubits,
                        'use_cuTensorNet_autotuning': True # Tensor network option
                    })
                    print(f"  blocking function enabled: blocking_qubits={blocking_qubits}")
                
                # cuStateVec available
                try:
                    test_sim = AerSimulator(method='statevector', device='GPU')
                    if hasattr(test_sim.options, 'cuStateVec_enable') and n_vars >= 5:
                        backend_options['cuStateVec_enable'] = True
                        print(f"  cuStateVec enabled")
                except:
                    pass
            
            backend = AerSimulator(**backend_options)
            backend.set_max_qubits(1000)
            print(f"  Avable Device: {available_devices}")
            print(f"  Selected Backend: {backend_options}")
            print(f"  Qubit Property Backend {backend.qubit_properties}")
            
            return backend
            
        except Exception as e:
            print(f"  Warning: Error in specified backend configuration: {e}")
            print(f"  Fallback: use default AerSimulator")
            return AerSimulator()

    def _solve_held_karp(self, **kwargs) -> TSPResult:
        """
        Solve TSP exactly using Held-Karp algorithm (dynamic programming)
        
        Satisfies the same constraints as OMMX Instance:
        1. At each time t, salesman is at exactly one city
        2. Each city is visited exactly once
        """
        start_time = time.time()
        n = self.n_cities
        
        # dp[mask][i] = minimum cost to visit cities in mask and end at city i
        # bit i in mask is 1 if city i has been visited
        dp = {}
        parent = {}
        
        # Initialize: start at city 0 at time 0 (same as OMMX Instance)
        dp[(1 << 0, 0)] = 0
        
        # Add one city at each time step (satisfies constraint 1)
        for t in range(1, n):
            # All combinations of visiting t+1 cities
            for mask in range(1 << n):
                if bin(mask).count('1') != t + 1:
                    continue
                
                # City 0 must be included (starting city)
                if not (mask & 1):
                    continue
                
                # List of cities in current set
                cities_in_mask = [i for i in range(n) if mask & (1 << i)]
                
                # City to visit at time t (last visited city)
                for last in cities_in_mask:
                    if last == 0 and t < n - 1:  # City 0 only at start
                        continue
                    
                    # Subset excluding last (state at time t-1)
                    prev_mask = mask ^ (1 << last)
                    
                    # Search for city at time t-1
                    min_cost = float('inf')
                    min_prev = -1
                    
                    for prev in range(n):
                        if not (prev_mask & (1 << prev)):
                            continue
                        if prev == last:
                            continue
                        
                        if (prev_mask, prev) in dp:
                            cost = dp[(prev_mask, prev)] + self.distance_matrix[prev][last]
                            if cost < min_cost:
                                min_cost = cost
                                min_prev = prev
                    
                    if min_cost < float('inf'):
                        dp[(mask, last)] = min_cost
                        parent[(mask, last)] = min_prev
        
        # Minimum cost to visit all cities and return to city 0 (satisfies constraint 2)
        full_mask = (1 << n) - 1
        min_cost = float('inf')
        last_city = -1
        
        # Move from city at time n-1 to city 0
        for i in range(1, n):
            if (full_mask, i) in dp:
                cost = dp[(full_mask, i)] + self.distance_matrix[i][0]
                if cost < min_cost:
                    min_cost = cost
                    last_city = i
        
        # Reconstruct path (in time order)
        path = []
        if last_city != -1 and min_cost < float('inf'):
            current = last_city
            mask = full_mask
            path = [current]
            
            # Trace back in reverse time order
            while (mask, current) in parent and parent[(mask, current)] != -1:
                next_city = parent[(mask, current)]
                path.append(next_city)
                mask ^= (1 << current)
                current = next_city
            
            path.reverse()
        
        # Check solution feasibility (same constraints as OMMX Instance)
        feasible = self._check_feasibility(path)
        
        computation_time = time.time() - start_time
        
        return TSPResult(
            path=path,
            distance=min_cost if min_cost < float('inf') else float('inf'),
            feasible=feasible,
            algorithm="held_karp",
            computation_time=computation_time,
            additional_info={
                'optimal': True,
                'num_states': len(dp),
                'satisfies_time_constraint': True,  # One city at each time
                'satisfies_location_constraint': True  # Each city visited once
            }
        )

    def _qubo_to_ising(self, Q: Union[np.ndarray, Dict], n_vars: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert QUBO matrix to Ising Hamiltonian"""
        if isinstance(Q, dict):
            Q_array = np.zeros((n_vars, n_vars))
            for (i, j), value in Q.items():
                if i < n_vars and j < n_vars:
                    Q_array[i, j] = value
        else:
            Q_array = Q
            n_vars = Q_array.shape[0]

        J = np.zeros((n_vars, n_vars))
        h = np.zeros(n_vars)
        offset = 0

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    h[i] -= Q_array[i, j] / 2
                    offset += Q_array[i, j] / 4
                elif i < j:
                    J[i, j] = Q_array[i, j] / 4 + Q_array[j, i] / 4
                    h[i] -= (Q_array[i, j] + Q_array[j, i]) / 4
                    h[j] -= (Q_array[i, j] + Q_array[j, i]) / 4
                    offset += (Q_array[i, j] + Q_array[j, i]) / 4

        return J, h, offset

    def _ising_to_pauli_hamiltonian(self, J: np.ndarray, h: np.ndarray) -> SparsePauliOp:
        """Convert Ising Hamiltonian to sum of Pauli operators"""
        n = len(h)
        pauli_list = []

        for i in range(n):
            for j in range(i+1, n):
                if abs(J[i, j]) > 1e-10:
                    pauli_str = ['I'] * n
                    pauli_str[i] = 'Z'
                    pauli_str[j] = 'Z'
                    pauli_list.append((''.join(reversed(pauli_str)), J[i, j]))

        for i in range(n):
            if abs(h[i]) > 1e-10:
                pauli_str = ['I'] * n
                pauli_str[i] = 'Z'
                pauli_list.append((''.join(reversed(pauli_str)), h[i]))

        return SparsePauliOp.from_list(pauli_list)

    def _extract_path(self, x_values: Dict[Tuple[int, ...], float]) -> List[int]:
        """Extract tour from decision variables"""
        path = []
        for t in range(self.n_cities):
            for i in range(self.n_cities):
                if x_values.get((t, i), 0) > 0.5:
                    path.append(i)
                    break
        return path

    def _extract_path_from_binary(self, x_values: List[int]) -> List[int]:
        """Extract tour from binary array"""
        N = self.n_cities
        path = []

        for t in range(N):
            for i in range(N):
                idx = t * N + i
                if idx < len(x_values) and x_values[idx] == 1:
                    path.append(i)
                    break

        return path

    def _check_feasibility_x(self, x_values: List[int]) -> bool:
        """Check solution feasibility"""
        N = self.n_cities
        # Check if at each time exactly one city is visited
        for t in range(N):
            count = sum(x_values[t * N + i] for i in range(N) if t * N + i < len(x_values))
            if count != 1:
                return False
        # Check if each city is visited exactly once
        for i in range(N):
            count = sum(x_values[t * N + i] for t in range(N) if t * N + i < len(x_values))
            if count != 1:
                return False

        return True
    
    def _check_feasibility(self, path: List[int]) -> bool:
        """
        Check if Held-Karp solution satisfies OMMX Instance constraints
        
        1. At each time, exactly one city is visited (path length equals n_cities)
        2. Each city is visited exactly once (no duplicates in path)
        """
        if len(path) != self.n_cities:
            return False
        
        # Check if each city appears exactly once
        if len(set(path)) != self.n_cities:
            return False
        
        # Check if all cities are included
        if set(path) != set(range(self.n_cities)):
            return False
        
        return True

    def evaluate_total_distance(self, path: List[int]) -> float:
        """Calculate total distance for given tour"""
        if len(path) != self.n_cities:
            return float('inf')

        total_distance = 0
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % len(path)]
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance

    def _evaluate_solution(self, result: TSPResult):
        """Perform detailed evaluation of solution"""
        if not result.feasible or not result.path:
            result.evaluation_metrics = {
                'valid_solution': False,
                'optimality_gap': float('inf'),
                'path_validity': False
            }
            return

        # Check path validity
        path_valid = self.evaluate_constraints(result.path)

        # Calculate evaluation metrics
        result.evaluation_metrics = {
            'valid_solution': result.feasible,
            'path_validity': path_valid['all_satisfied'],
            'total_distance': result.distance,
            'average_edge_distance': result.distance / self.n_cities if path_valid else float('inf'),
            'computation_time': result.computation_time
        }

        # Compare with optimal solution (if already computed)
        optimal_result = self._get_optimal_solution()
        if optimal_result:
            gap = abs(result.distance - optimal_result.distance) / optimal_result.distance * 100 if optimal_result.distance > 0 else 0
            result.evaluation_metrics['optimality_gap'] = gap
            result.evaluation_metrics['gap_to_optimal'] = abs(result.distance - optimal_result.distance)

    def evaluate_constraints(self, path: List[int]) -> Dict[str, bool]:
        """
        Evaluate if given tour satisfies OMMX Instance constraints
        
        Returns
        -------
        dict
            'time_constraint': Whether at each time exactly one city is visited
            'location_constraint': Whether each city is visited exactly once
            'all_satisfied': Whether all constraints are satisfied
        """
        time_constraint = len(path) == self.n_cities
        location_constraint = (len(set(path)) == self.n_cities and 
                             set(path) == set(range(self.n_cities)) and
                             all(0 <= city < self.n_cities for city in path))
        
        return {
            'time_constraint': time_constraint,
            'location_constraint': location_constraint,
            'all_satisfied': time_constraint and location_constraint
        }
    
    def _get_optimal_solution(self) -> Optional[TSPResult]:
        """Get optimal solution (Held-Karp) from history"""
        for result in self.results_history:
            if result.algorithm == "held_karp" and result.feasible:
                return result
        return None

    # ================================
    # Enhanced comparison features (Task 4)
    # ================================

    def export_comparison_table(self, output_format: str = 'all', prefix: str = None, output_dir: str = None) -> Dict[str, str]:
        """
        Export algorithm comparison results in table format
        
        Parameters
        ----------
        output_format : str
            Output format ('csv', 'excel', 'html', 'json', 'all')
        prefix : str, optional
            Filename prefix
        output_dir : str, optional
            Output directory (current directory if not specified)
        
        Returns
        -------
        dict
            Dictionary of generated file paths
        """
        if not self.results_history:
            print("No results to export.")
            return {}

        # Set output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path('.')

        # Use current timestamp if prefix not specified
        if prefix is None:
            prefix = f"tsp_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Prepare comparison data
        comparison_data = []
        optimal_result = self._get_optimal_solution()

        for result in self.results_history:
            row = {
                'Algorithm': result.algorithm.upper(),
                'Total Distance': result.distance,
                'Computation Time (s)': result.computation_time,
                'Feasible': 'Yes' if result.feasible else 'No',
                'Path Valid': 'Yes' if result.evaluation_metrics.get('path_validity', False) else 'No',
                'Cities Visited': len(result.path),
                'Unique Cities': len(set(result.path)) if result.path else 0,
            }

            # Compare with optimal solution
            if optimal_result and result.distance != float('inf'):
                gap = result.evaluation_metrics.get('optimality_gap', float('inf'))
                row['Optimality Gap (%)'] = f"{gap:.2f}" if gap != float('inf') else 'N/A'
                gap_to_optimal = result.evaluation_metrics.get('gap_to_optimal', float('inf'))
                row['Gap to Optimal'] = f"{gap_to_optimal:.5f}" if gap_to_optimal != float('inf') else 'N/A'
            else:
                row['Optimality Gap (%)'] = 'N/A'
                row['Gap to Optimal'] = 'N/A'

            # Algorithm-specific information
            if result.algorithm == "simulated_annealing":
                row['SA Reads'] = result.additional_info.get('num_reads', 'N/A')
                row['SA Feasible Solutions'] = result.additional_info.get('num_feasible_solutions', 'N/A')
            elif result.algorithm == "qaoa":
                row['QAOA Layers (p)'] = result.additional_info.get('p', 'N/A')
                row['QAOA Iterations'] = result.additional_info.get('iterations', 'N/A')
                row['QAOA Shots'] = result.additional_info.get('shots', 'N/A')
            elif result.algorithm == "held_karp":
                row['DP States'] = result.additional_info.get('num_states', 'N/A')

            comparison_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Output file paths
        output_files = {}

        # CSV output
        if output_format in ['csv', 'all']:
            csv_path = output_path / f"{prefix}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            output_files['csv'] = str(csv_path)
            print(f"CSV file saved: {csv_path}")

        # Excel output
        if output_format in ['excel', 'all']:
            excel_path = output_path / f"{prefix}.xlsx"
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Comparison', index=False)
                
                # Format settings
                workbook = writer.book
                worksheet = writer.sheets['Comparison']
                
                # Header format
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D7E4BD',
                    'border': 1
                })
                
                # Auto-adjust column width
                for i, col in enumerate(df.columns):
                    column_len = max(df[col].astype(str).str.len().max(), len(col))
                    worksheet.set_column(i, i, column_len + 2)
                
                # Conditional formatting (highlight feasible solutions)
                worksheet.conditional_format(1, 0, len(df), len(df.columns) - 1, {
                    'type': 'formula',
                    'criteria': '=$D2="Yes"',
                    'format': workbook.add_format({'bg_color': '#C6EFCE'})
                })
            
            output_files['excel'] = str(excel_path)
            print(f"Excel file saved: {excel_path}")

        # HTML output
        if output_format in ['html', 'all']:
            html_path = output_path / f"{prefix}.html"
            # Find fastest algorithm(s)
            min_time = min(r.computation_time for r in self.results_history)
            fastest_solutions = [r for r in self.results_history if r.computation_time == min_time]
            min_distance = min(r.distance for r in self.results_history)
            best_solutions = [r for r in self.results_history if r.distance == min_distance]
            best_algorithms = [r.algorithm for r in best_solutions]
            fastest_str = ""
            if len(fastest_solutions) == 1:
                fastest_str = f"{fastest_solutions[0].algorithm.upper()} ({min_time:.5f}s)"
            else:
                algorithms_str = ", ".join([r.algorithm.upper() for r in fastest_solutions])
                fastest_str = f"{algorithms_str} ({min_time:.5f}s)"
            if len(best_solutions) == 1:
                best_solution_str = f"{best_solutions[0].algorithm.upper()} ({min_distance:.5f}s)"
            else:
                algorithms_str = ", ".join([r.algorithm.upper() for r in best_solutions])
                best_solution_str = f"{algorithms_str} ({min_distance:.5f}s)"
            
            html_content = f"""
            <html>
            <head>
                <title>TSP Algorithm Comparison</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .feasible {{ background-color: #C6EFCE; }}
                    .infeasible {{ background-color: #FFC7CE; }}
                    .summary {{ margin-top: 20px; padding: 10px; background-color: #f0f0f0; }}
                </style>
            </head>
            <body>
                <h1>TSP Algorithm Comparison Results</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Number of cities: {self.n_cities}</p>
                
                {df.to_html(classes='comparison-table', index=False)}
                
                <div class="summary">
                    <h2>Summary</h2>
                    <ul>
                        <li>Total algorithms tested: {len(self.results_history)}</li>
                        <li>Feasible solutions found: {sum(1 for r in self.results_history if r.feasible)}</li>
                        <li>Best solution(s): {best_solution_str}</li>
                        <li>Fastest algorithm(s): {fastest_str}</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            output_files['html'] = str(html_path)
            print(f"HTML file saved: {html_path}")

        # JSON output
        if output_format in ['json', 'all']:
            json_path = output_path / f"{prefix}.json"
            
            # Convert comparison_data values to JSON serializable format
            json_safe_comparison = []
            for row in comparison_data:
                safe_row = {}
                for key, value in row.items():
                    if isinstance(value, (np.ndarray, np.generic)):
                        safe_row[key] = value.tolist()
                    elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                        safe_row[key] = 'inf' if np.isinf(value) else 'nan'
                    else:
                        safe_row[key] = value
                json_safe_comparison.append(safe_row)
            
            # Find best solutions
            min_distance = min(r.distance for r in self.results_history)
            best_solutions = [r for r in self.results_history if r.distance == min_distance]
            best_algorithms = [r.algorithm for r in best_solutions]
            
            # Find fastest algorithm(s)
            min_time = min(r.computation_time for r in self.results_history)
            fastest_solutions = [r for r in self.results_history if r.computation_time == min_time]
            fastest_algorithms = [r.algorithm for r in fastest_solutions]
            
            json_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'n_cities': self.n_cities,
                    'algorithms_tested': [r.algorithm for r in self.results_history],
                    'best_algorithms': best_algorithms,
                    'best_distance': min_distance,
                    'fastest_algorithms': fastest_algorithms,
                    'fastest_time': min_time
                },
                'results': [r.to_dict() for r in self.results_history],
                'comparison': json_safe_comparison
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            output_files['json'] = str(json_path)
            print(f"JSON file saved: {json_path}")

        return output_files

    def display_comparison_summary(self):
        """
        Display algorithm comparison summary
        Analyze feasibility and computation time vs solution quality trade-offs
        """
        if not self.results_history:
            print("No results to compare.")
            return

        print("\n" + "="*80)
        print("TSP ALGORITHM COMPARISON SUMMARY")
        print("="*80)
        print(f"Problem size: {self.n_cities} cities")
        print(f"Total algorithms tested: {len(self.results_history)}")
        print()

        # Feasibility analysis
        print("FEASIBILITY ANALYSIS:")
        print("-"*40)
        for result in self.results_history:
            feasible_status = "✓ Feasible" if result.feasible else "✗ Infeasible"
            constraints = self.evaluate_constraints(result.path)
            print(f"{result.algorithm.upper():<20} {feasible_status:<15} "
                  f"Time constraint: {'✓' if constraints['time_constraint'] else '✗'}  "
                  f"Location constraint: {'✓' if constraints['location_constraint'] else '✗'}")
        print()

        # Comparison of feasible solutions only
        feasible_results = [r for r in self.results_history if r.feasible]
        if feasible_results:
            print("FEASIBLE SOLUTIONS COMPARISON:")
            print("-"*40)
            print(f"{'Algorithm':<20} {'Distance':<15} {'Time (s)':<15} {'Gap to Optimal':<15}")
            
            optimal_result = self._get_optimal_solution()
            for result in sorted(feasible_results, key=lambda x: x.distance):
                gap_str = ""
                if optimal_result and result.distance != float('inf'):
                    gap = result.evaluation_metrics.get('optimality_gap', 0)
                    gap_str = f"{gap:.2f}%" if gap < float('inf') else "N/A"
                
                print(f"{result.algorithm.upper():<20} {result.distance:<15.5f} "
                      f"{result.computation_time:<15.5f} {gap_str:<15}")
        print()

        # Trade-off analysis
        print("PERFORMANCE TRADE-OFF ANALYSIS:")
        print("-"*40)
        
        # Fastest algorithm(s)
        min_time = min(r.computation_time for r in self.results_history)
        fastest_solutions = [r for r in self.results_history if r.computation_time == min_time]
        
        if len(fastest_solutions) == 1:
            print(f"Fastest algorithm: {fastest_solutions[0].algorithm.upper()} ({min_time:.5f}s)")
        else:
            algorithms_str = ", ".join([r.algorithm.upper() for r in fastest_solutions])
            print(f"Fastest algorithms (tied): {algorithms_str} ({min_time:.5f}s)")
        
        # Best solution(s) (feasible only)
        if feasible_results:
            min_distance = min(r.distance for r in feasible_results)
            best_solutions = [r for r in feasible_results if r.distance == min_distance]
            
            if len(best_solutions) == 1:
                print(f"Best solution: {best_solutions[0].algorithm.upper()} (distance: {min_distance:.5f})")
            else:
                # Multiple algorithms achieved the same optimal distance
                algorithms_str = ", ".join([r.algorithm.upper() for r in best_solutions])
                print(f"Best solutions (tied): {algorithms_str} (distance: {min_distance:.5f})")
            
            # Efficiency score (distance/time)
            print("\nEfficiency Score (upper is better - distance/time):")
            efficiency_scores = []
            for result in feasible_results:
                if result.computation_time > 0:
                    efficiency = result.distance / result.computation_time
                    efficiency_scores.append((result.algorithm, efficiency))
                    print(f"  {result.algorithm.upper()}: {efficiency:.2f}")
            
            # Highlight best efficiency
            if efficiency_scores:
                min_efficiency = max(score[1] for score in efficiency_scores)
                best_efficient = [algo for algo, eff in efficiency_scores if eff == min_efficiency]
                if len(best_efficient) == 1:
                    print(f"\nMost efficient: {best_efficient[0].upper()}")
                else:
                    print(f"\nMost efficient (tied): {', '.join([a.upper() for a in best_efficient])}")
        
        print("\n" + "="*80)

    # ================================
    # Visualization functions included in library
    # ================================

    def visualize_cities(self, title: str = "TSP Cities", save_path: str = None, output_dir: str = None):
        """
        Visualize city locations (supports estimated coordinates)
        
        Parameters
        ----------
        title : str
            Graph title
        save_path : str, optional
            Save filename (including extension)
        output_dir : str, optional
            Output directory
        """
        if self.coordinates is None:
            print("Coordinates are not available for visualization")
            return

        plt.figure(figsize=(8, 6))
        x_coords = [c[0] for c in self.coordinates]
        y_coords = [c[1] for c in self.coordinates]
        plt.scatter(x_coords, y_coords, c='red', s=200, zorder=5)

        for i, (x, y) in enumerate(self.coordinates):
            plt.annotate(str(i), (x, y), ha='center', va='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                                 edgecolor='black', alpha=0.8),
                        zorder=10)

        # Add note if coordinates are estimated
        if self.coordinates_estimated:
            title += " (Coordinates estimated from distance matrix)"

        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)

        if save_path:
            # Set output path
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                full_path = output_path / save_path
            else:
                full_path = Path(save_path)
                
            plt.savefig(full_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Cities visualization saved to {full_path}")
        else:
            plt.show()
            plt.close()

    def visualize_solution(self, result: TSPResult, title: Optional[str] = None, save_path: Optional[str] = None, output_dir: str = None):
        """
        Visualize TSP solution with detailed information (supports estimated coordinates)
        
        Parameters
        ----------
        result : TSPResult
            Solution to visualize
        title : str, optional
            Graph title
        save_path : str, optional
            Save filename
        output_dir : str, optional
            Output directory
        """
        if self.coordinates is None:
            print("Coordinates are not available for visualization")
            return

        if title is None:
            title = f"TSP Solution - {result.algorithm.upper()}"

        path = result.path
        n_cities = len(self.coordinates)

        plt.figure(figsize=(10, 8))

        # Plot cities
        x_coords = [self.coordinates[i][0] for i in range(n_cities)]
        y_coords = [self.coordinates[i][1] for i in range(n_cities)]
        plt.scatter(x_coords, y_coords, c='red', s=200, zorder=5)

        # Display city numbers
        for i in range(n_cities):
            x, y = self.coordinates[i]
            plt.annotate(str(i), (x, y), ha='center', va='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                                 edgecolor='black', alpha=0.8),
                        zorder=10)

        # Draw tour
        if path and len(path) == n_cities:
            path_with_return = path + [path[0]]
            x_path = [self.coordinates[i][0] for i in path_with_return]
            y_path = [self.coordinates[i][1] for i in path_with_return]
            plt.plot(x_path, y_path, 'b-', linewidth=2, alpha=0.7)

            # Add arrows to show direction
            for i in range(len(path)):
                start = self.coordinates[path[i]]
                end = self.coordinates[path[(i+1) % len(path)]]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                plt.arrow(start[0], start[1], dx*0.8, dy*0.8,
                         head_width=0.02, head_length=0.03, fc='blue', ec='blue', alpha=0.6)

        # Add detailed information
        info_text = f"Distance: {result.distance:.5f}\n"
        info_text += f"Time: {result.computation_time:.5f}s\n"
        info_text += f"Feasible: {'Yes' if result.feasible else 'No'}"

        if 'optimality_gap' in result.evaluation_metrics:
            gap = result.evaluation_metrics['optimality_gap']
            if gap != float('inf'):
                info_text += f"\nGap: {gap:.2f}%"

        # Add note if coordinates are estimated
        if self.coordinates_estimated:
            title += " (Estimated coordinates)"
            info_text += "\nCoords: Estimated"

        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()

        if save_path:
            # Set output path
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                full_path = output_path / save_path
            else:
                full_path = Path(save_path)
                
            plt.savefig(full_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Solution visualization saved to {full_path}")
        else:
            plt.show()
            plt.close()

    def plot_comparison(self, save_path: str = "algorithm_comparison.png", output_dir: str = None):
        """
        Visualize algorithm comparison
        
        Parameters
        ----------
        save_path : str
            Save filename
        output_dir : str, optional
            Output directory
        """
        if not self.results_history:
            print("No results to compare.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Distance comparison
        algorithms = [r.algorithm for r in self.results_history]
        distances = [r.distance for r in self.results_history]

        ax = axes[0, 0]
        bars = ax.bar(algorithms, distances)
        ax.set_ylabel('Total Distance')
        ax.set_title('Distance Comparison')

        # Highlight all optimal solutions
        min_distance = min(distances)
        for i, (algo, dist) in enumerate(zip(algorithms, distances)):
            if dist == min_distance:
                bars[i].set_color('green')

        # Computation time comparison
        ax = axes[0, 1]
        times = [r.computation_time for r in self.results_history]
        time_bars = ax.bar(algorithms, times)
        ax.set_ylabel('Computation Time (s)')
        ax.set_title('Computation Time Comparison')
        ax.set_yscale('log')  # Log scale
        
        # Highlight fastest algorithm(s)
        min_time = min(times)
        for i, time in enumerate(times):
            if time == min_time:
                time_bars[i].set_color('orange')

        # Optimality gap
        ax = axes[1, 0]
        gaps = []
        for r in self.results_history:
            gap = r.evaluation_metrics.get('optimality_gap', 0)
            gaps.append(gap if gap != float('inf') else 0)

        gap_bars = ax.bar(algorithms, gaps)
        ax.set_ylabel('Optimality Gap (%)')
        ax.set_title('Optimality Gap')
        
        # Highlight optimal solutions (0% gap)
        for i, gap in enumerate(gaps):
            if gap == 0:
                gap_bars[i].set_color('green')

        # Feasibility
        ax = axes[1, 1]
        feasible_counts = [1 if r.feasible else 0 for r in self.results_history]
        feasibility_bars = ax.bar(algorithms, feasible_counts)
        ax.set_ylabel('Feasible (1) / Infeasible (0)')
        ax.set_title('Solution Feasibility')
        ax.set_ylim(0, 1.5)
        
        # Color code feasibility
        for i, is_feasible in enumerate(feasible_counts):
            if is_feasible:
                feasibility_bars[i].set_color('green')
            else:
                feasibility_bars[i].set_color('red')

        plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave space for bottom text
        
        # Add a general note about color coding
        fig.text(0.5, 0.01, 'Green: best distance/optimal • Orange: fastest • Red: infeasible', 
                ha='center', fontsize=10, style='italic')
        
        # Set output path
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            full_path = output_path / save_path
        else:
            full_path = Path(save_path)
            
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison chart saved to {full_path}")

    def plot_qaoa_convergence(self, result: Optional[TSPResult] = None, save_path: str = "qaoa_convergence.png", output_dir: str = None):
        """
        Visualize QAOA optimization convergence
        
        Parameters
        ----------
        result : TSPResult, optional
            QAOA result (get from history if not specified)
        save_path : str
            Save filename
        output_dir : str, optional
            Output directory
        """
        if result is None:
            # Find QAOA result from history
            qaoa_results = [r for r in self.results_history if r.algorithm == "qaoa"]
            if not qaoa_results:
                print("No QAOA results found.")
                return
            result = qaoa_results[-1]

        if 'cost_history' not in result.additional_info or not result.additional_info['cost_history']:
            print("QAOA cost history is not available.")
            return

        plt.figure(figsize=(10, 6))
        cost_history = result.additional_info['cost_history']
        plt.plot(cost_history, linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function Value')
        plt.title('QAOA Optimization Convergence')
        plt.grid(True, alpha=0.3)

        # Highlight minimum
        min_cost = min(cost_history)
        min_idx = cost_history.index(min_cost)
        plt.axhline(y=min_cost, color='r', linestyle='--', alpha=0.5)
        plt.annotate(f'Min: {min_cost:.5f} at iter {min_idx}',
                    xy=(min_idx, min_cost),
                    xytext=(min_idx + 5, min_cost + 0.5),
                    arrowprops=dict(arrowstyle='->', color='red'))

        plt.tight_layout()
        
        # Set output path
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            full_path = output_path / save_path
        else:
            full_path = Path(save_path)
            
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"QAOA convergence chart saved to {full_path}")

    # ================================
    # Comparison function avoiding recalculation
    # ================================

    def compare_algorithms(self, algorithms: List[Union[Algorithm, str]] = None, **kwargs) -> pd.DataFrame:
        """Compare multiple algorithms (avoiding recalculation)"""
        if algorithms is None:
            algorithms = [Algorithm.SIMULATED_ANNEALING, Algorithm.QAOA, Algorithm.HELD_KARP]

        # Check already executed algorithms
        executed_algorithms = {r.algorithm for r in self.results_history}

        for algo in algorithms:
            algo_name = algo.value if isinstance(algo, Algorithm) else algo
            if algo_name not in executed_algorithms:
                print(f"\nExecuting: {algo_name}")
                result = self.solve(algo, **kwargs)
            else:
                print(f"\n{algo_name} already executed.")

        # Convert results to DataFrame
        comparison_data = []
        optimal_result = self._get_optimal_solution()

        for result in self.results_history:
            data = {
                'Algorithm': result.algorithm,
                'Distance': result.distance,
                'Computation Time (s)': result.computation_time,
                'Feasible': result.feasible,
                'Valid Path': result.evaluation_metrics.get('path_validity', False)
            }

            if optimal_result:
                gap = result.evaluation_metrics.get('optimality_gap', float('inf'))
                data['Optimality Gap (%)'] = gap if gap != float('inf') else 'N/A'
                gap_to_optimal = result.evaluation_metrics.get('gap_to_optimal', float('inf'))
                data['Gap to Optimal'] = gap_to_optimal if gap_to_optimal != float('inf') else 'N/A'

            comparison_data.append(data)

        df = pd.DataFrame(comparison_data)
        return df

    def generate_report(self, save_path: str = "tsp_benchmark_report.txt", output_dir: str = None):
        """
        Generate benchmark results report
        
        Parameters
        ----------
        save_path : str
            Save filename
        output_dir : str, optional
            Output directory
        """
        # Set output path
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            full_path = output_path / save_path
        else:
            full_path = Path(save_path)
            
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write("=== TSP Benchmark Report (Qiskit v2.0 Compatible) ===\n\n")
            f.write(f"Problem size: {self.n_cities} cities\n")
            if self.coordinates_estimated:
                f.write("Coordinates: Estimated from distance matrix (using MDS)\n")
            f.write("\n")

            # Algorithm comparison
            if self.results_history:
                f.write("Algorithm Comparison Results:\n")
                f.write("-" * 80 + "\n")

                # Sort results by distance
                sorted_results = sorted(self.results_history, key=lambda x: x.distance)

                for i, result in enumerate(sorted_results):
                    f.write(f"\n{i+1}. {result.algorithm.upper()}\n")
                    f.write(f"   Total distance: {result.distance:.5f}\n")
                    f.write(f"   Computation time: {result.computation_time:.5f}s\n")
                    f.write(f"   Feasible solution: {'Yes' if result.feasible else 'No'}\n")

                    if 'optimality_gap' in result.evaluation_metrics:
                        gap = result.evaluation_metrics['optimality_gap']
                        f.write(f"   Optimality gap: {gap:.2f}%\n" if gap < float('inf') else "   Optimality gap: N/A\n")

                    f.write(f"   Path: {result.path}\n")
                    constraints = self.evaluate_constraints(result.path)
                    f.write(f"   Constraint check: {'✓ All satisfied' if constraints['all_satisfied'] else '✗ Violated'}\n")

                    if result.algorithm == "qaoa":
                        f.write(f"   QAOA additional info:\n")
                        f.write(f"     - Layers (p): {result.additional_info.get('p', 'N/A')}\n")
                        f.write(f"     - Optimizer: {result.additional_info.get('optimizer', 'N/A')}\n")
                        f.write(f"     - Iterations: {result.additional_info.get('iterations', 'N/A')}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Improvements (Qiskit v2.0 compatible):\n")
            f.write("- Using scipy.optimize.minimize directly instead of qiskit_algorithms\n")
            f.write("- Using EstimatorV2, SamplerV2 primitives\n")
            f.write("- Improved error handling and fallback functionality\n")
            f.write("- Detailed evaluation and comparison of feasible solutions\n")
            f.write("- Multiple format result output functionality\n")

        print(f"Report saved to {full_path}")
        
    def visualize_qaoa_circuit(self, result: Optional[TSPResult] = None, 
                          save_path: str = "qaoa_circuit.png", 
                          output_dir: str = None,
                          decompose_level: int = 0,
                          show_layout: bool = True,
                          show_depth: bool = True,
                          split_files: bool = True,
                          max_width: int = 30):
        """
        Visualize QAOA quantum circuit
        
        Parameters
        ----------
        result : TSPResult, optional
            QAOA result (get from history if not specified)
        save_path : str
            Save filename for circuit diagram (base name if split_files=True)
        output_dir : str, optional
            Output directory
        decompose_level : int
            Level of circuit decomposition (0: no decomposition, 1+: decompose gates)
        show_layout : bool
            Whether to show qubit layout information
        show_depth : bool
            Whether to show circuit depth
        split_files : bool
            Whether to save circuits in separate files for better visibility
        max_width : int
            Maximum width before folding the circuit (default: 30)
        """
        if result is None:
            # Find QAOA result from history
            qaoa_results = [r for r in self.results_history if r.algorithm == "qaoa"]
            if not qaoa_results:
                print("No QAOA results found.")
                return
            result = qaoa_results[-1]
        
        if result.algorithm != "qaoa":
            print("This result is not from QAOA algorithm.")
            return
        
        # Get circuit from class instance
        if not hasattr(self, '_last_qaoa_circuit') or self._last_qaoa_circuit is None:
            print("Circuit information not available. Re-run QAOA to capture circuit.")
            return
        
        circuit_info = self._last_qaoa_circuit
        
        # Debug information for Qiskit v2.0
        print(f"Available circuits in _last_qaoa_circuit: {list(circuit_info.keys())}")
        
        # Check what circuits are available
        has_original = 'original' in circuit_info and circuit_info['original'] is not None
        has_transpiled = 'transpiled' in circuit_info and circuit_info['transpiled'] is not None
        has_bound_original = 'bound_original' in circuit_info and circuit_info['bound_original'] is not None
        has_bound_transpiled = 'bound_transpiled' in circuit_info and circuit_info['bound_transpiled'] is not None
        
        # Import circuit drawer
        from qiskit.visualization import circuit_drawer
        
        # Set output path
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path('.')
        
        # Get base filename without extension
        base_name = Path(save_path).stem
        extension = Path(save_path).suffix or '.png'
        
        saved_files = []
        
        if split_files:
            # Save each circuit in a separate file for better visibility
            
            # 1. Original or bound original circuit
            if has_original or has_bound_original:
                circuit = circuit_info.get('bound_original', circuit_info.get('original'))
                circuit_type = "bound_original" if has_bound_original else "original"
                
                # Calculate optimal fold value based on circuit depth
                circuit_depth = circuit.depth()
                fold_value = max_width  # Fold at specified width
                
                # Calculate figure size based on folding
                if circuit_depth > fold_value:
                    # Multi-row layout
                    num_rows = (circuit_depth + fold_value - 1) // fold_value
                    fig_width = min(25, fold_value * 0.5)
                    fig_height = max(8, num_rows * circuit.num_qubits * 0.3)
                else:
                    # Single row layout
                    fig_width = min(25, circuit_depth * 0.5)
                    fig_height = max(8, circuit.num_qubits * 0.5)
                
                fig = plt.figure(figsize=(fig_width, fig_height))
                
                try:
                    # Draw circuit with folding for better visibility
                    circuit.draw("mpl")
                    
                    title_type = "Bound Original" if has_bound_original else "Original (Parametrized)"
                    plt.suptitle(f"QAOA {title_type} Circuit\n(p={result.additional_info.get('p', 'N/A')}, "
                                f"{circuit.num_qubits} qubits, depth={circuit_depth})", 
                                fontsize=16, y=0.98)
                    
                    # Add fold information if circuit was folded
                    if circuit_depth > fold_value:
                        plt.figtext(0.5, 0.01, f"Circuit folded at width {fold_value}", 
                                ha='center', fontsize=10, style='italic')
                    
                    # Save with high DPI
                    file_path = output_path / f"{base_name}_{circuit_type}{extension}"
                    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
                    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
                    plt.close()
                    saved_files.append(str(file_path))
                    print(f"Saved {circuit_type} circuit to {file_path} (fold={fold_value})")
                    
                except Exception as e:
                    print(f"Error drawing {circuit_type} circuit: {e}")
                    plt.close()
            
            # 2. Transpiled or bound transpiled circuit
            if has_transpiled or has_bound_transpiled:
                transpiled_circuit = circuit_info.get('bound_transpiled', circuit_info.get('transpiled'))
                circuit_type = "bound_transpiled" if has_bound_transpiled else "transpiled"
                
                if decompose_level > 0 and transpiled_circuit is not None:
                    for _ in range(decompose_level):
                        transpiled_circuit = transpiled_circuit.decompose()
                
                # Transpiled circuits often have much higher depth
                circuit_depth = transpiled_circuit.depth()
                
                # Adaptive fold value for transpiled circuits
                if circuit_depth > 200:
                    fold_value = 50  # More aggressive folding for very deep circuits
                elif circuit_depth > 100:
                    fold_value = 40
                else:
                    fold_value = max_width
                
                # Calculate figure size based on folding
                if circuit_depth > fold_value:
                    num_rows = (circuit_depth + fold_value - 1) // fold_value
                    fig_width = min(30, fold_value * 0.4)
                    fig_height = max(10, num_rows * transpiled_circuit.num_qubits * 0.25)
                else:
                    fig_width = min(30, circuit_depth * 0.4)
                    fig_height = max(10, transpiled_circuit.num_qubits * 0.4)
                
                fig = plt.figure(figsize=(fig_width, fig_height))
                
                try:
                    # Draw with adaptive style for transpiled circuits
                    circuit_drawer(transpiled_circuit, 
                                output='mpl',
                                style={
                                    'fontsize': 12,
                                    'subfontsize': 10,
                                    'displaytext': {'fontsize': 11},
                                    'displaycolor': {'fontsize': 11},
                                    'gatefacecolor': 'white',
                                    'barrierfacecolor': 'lightgray',
                                    'compress': True  # Compress gate spacing
                                },
                                fold=fold_value,
                                plot_barriers=False,  # Hide barriers for cleaner look
                                initial_state=False,
                                cregbundle=True,
                                reverse_bits=False,
                                justify='left')  # Left justify for better alignment
                    
                    title_type = "Bound Transpiled" if has_bound_transpiled else "Transpiled (Parametrized)"
                    plt.suptitle(f"{title_type} Circuit\n(depth={circuit_depth}, folded at {fold_value})", 
                                fontsize=16, y=0.98)
                    
                    # Add optimization level info
                    opt_level = circuit_info.get('optimization_level', 'N/A')
                    plt.figtext(0.5, 0.01, f"Optimization Level: {opt_level}", 
                            ha='center', fontsize=10, style='italic')
                    
                    # Save with high DPI
                    file_path = output_path / f"{base_name}_{circuit_type}{extension}"
                    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
                    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
                    plt.close()
                    saved_files.append(str(file_path))
                    print(f"Saved {circuit_type} circuit to {file_path} (fold={fold_value})")
                    
                except Exception as e:
                    print(f"Error drawing {circuit_type} circuit: {e}")
                    plt.close()
            
            # 3. Circuit statistics in a separate image (same as before)
            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
            ax.axis('off')
            
            # Collect circuit statistics with enhanced formatting
            stats_text = "QAOA Circuit Statistics\n" + "="*60 + "\n\n"
            
            # Problem information
            stats_text += f"Problem Size: {self.n_cities} cities ({self.n_cities**2} qubits)\n\n"
            
            # Original circuit stats
            if has_original or has_bound_original:
                orig = circuit_info.get('bound_original', circuit_info.get('original'))
                stats_text += "Original Circuit:\n"
                stats_text += "-"*30 + "\n"
                stats_text += f"  Dimensions:\n"
                stats_text += f"    - Qubits: {orig.num_qubits}\n"
                stats_text += f"    - Total Gates: {orig.size()}\n"
                stats_text += f"    - Circuit Depth: {orig.depth()}\n"
                stats_text += f"    - Parameters: {orig.num_parameters}\n"
                stats_text += f"    - Parameter Bound: {'Yes' if has_bound_original else 'No'}\n"
                
                # Gate breakdown with better formatting
                gate_counts = dict(orig.count_ops())
                if gate_counts:
                    stats_text += f"\n  Gate Breakdown:\n"
                    max_gate_name_len = max(len(gate) for gate in gate_counts.keys())
                    for gate, count in sorted(gate_counts.items(), key=lambda x: x[1], reverse=True):
                        stats_text += f"    {gate:<{max_gate_name_len}} : {count:>4}\n"
                stats_text += "\n"
            
            # Transpiled circuit stats
            if has_transpiled or has_bound_transpiled:
                trans = circuit_info.get('bound_transpiled', circuit_info.get('transpiled'))
                stats_text += "Transpiled Circuit:\n"
                stats_text += "-"*30 + "\n"
                stats_text += f"  Dimensions:\n"
                stats_text += f"    - Total Gates: {trans.size()}\n"
                stats_text += f"    - Circuit Depth: {trans.depth()}\n"
                stats_text += f"    - Optimization Level: {circuit_info.get('optimization_level', 'N/A')}\n"
                stats_text += f"    - Parameter Bound: {'Yes' if has_bound_transpiled else 'No'}\n"
                
                # Gate breakdown for transpiled circuit
                gate_counts = dict(trans.count_ops())
                if gate_counts:
                    stats_text += f"\n  Basis Gate Breakdown:\n"
                    # Group by gate type
                    single_qubit_gates = {}
                    two_qubit_gates = {}
                    other_gates = {}
                    
                    for gate, count in gate_counts.items():
                        if gate in ['rx', 'ry', 'rz', 'x', 'y', 'z', 'h', 's', 't', 'sx', 'sxdg']:
                            single_qubit_gates[gate] = count
                        elif gate in ['cx', 'cy', 'cz', 'ch', 'swap', 'iswap', 'ecr', 'rzz']:
                            two_qubit_gates[gate] = count
                        else:
                            other_gates[gate] = count
                    
                    if single_qubit_gates:
                        stats_text += "    Single-qubit gates:\n"
                        for gate, count in sorted(single_qubit_gates.items()):
                            stats_text += f"      {gate:<6} : {count:>4}\n"
                    
                    if two_qubit_gates:
                        stats_text += "    Two-qubit gates:\n"
                        for gate, count in sorted(two_qubit_gates.items()):
                            stats_text += f"      {gate:<6} : {count:>4}\n"
                    
                    if other_gates:
                        stats_text += "    Other gates:\n"
                        for gate, count in sorted(other_gates.items()):
                            stats_text += f"      {gate:<6} : {count:>4}\n"
                stats_text += "\n"
            
            # QAOA configuration
            stats_text += "QAOA Configuration:\n"
            stats_text += "-"*30 + "\n"
            stats_text += f"  Algorithm Parameters:\n"
            stats_text += f"    - QAOA Layers (p): {result.additional_info.get('p', 'N/A')}\n"
            stats_text += f"    - Optimizer: {result.additional_info.get('optimizer', 'N/A')}\n"
            stats_text += f"    - Iterations: {result.additional_info.get('iterations', 'N/A')}\n"
            stats_text += f"    - Shots: {result.additional_info.get('shots', 'N/A')}\n"
            
            # Handle final_cost formatting
            final_cost = result.additional_info.get('final_cost', 'N/A')
            if isinstance(final_cost, (int, float)) and final_cost != float('inf'):
                stats_text += f"    - Final Cost: {final_cost:.6f}\n"
            else:
                stats_text += f"    - Final Cost: {final_cost}\n"
            
            # Add optimal parameters info with nice formatting
            if 'optimal_params' in circuit_info:
                params = circuit_info['optimal_params']
                if hasattr(params, '__len__') and len(params) > 0:
                    stats_text += f"\n  Optimal Parameters ({len(params)} values):\n"
                    p_value = result.additional_info.get('p', len(params)//2)
                    
                    # Format parameters in columns
                    beta_params = params[:p_value]
                    gamma_params = params[p_value:]
                    
                    stats_text += "    β (mixer) parameters:\n"
                    for i, beta in enumerate(beta_params):
                        stats_text += f"      β[{i}] = {beta:>10.6f} ({beta/np.pi:.3f}π)\n"
                    
                    stats_text += "    γ (cost) parameters:\n"
                    for i, gamma in enumerate(gamma_params):
                        stats_text += f"      γ[{i}] = {gamma:>10.6f} ({gamma/np.pi:.3f}π)\n"
            
            # Environment info
            stats_text += "\nEnvironment:\n"
            stats_text += "-"*30 + "\n"
            try:
                import qiskit
                stats_text += f"  - Qiskit Version: {qiskit.__version__}\n"
            except:
                pass
            stats_text += f"  - Backend: {result.additional_info.get('backend_name', 'N/A')}\n"
            stats_text += f"  - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Draw text with monospace font
            ax.text(0.05, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=11,
                linespacing=1.5)
            
            # Save statistics
            file_path = output_path / f"{base_name}_statistics{extension}"
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()
            saved_files.append(str(file_path))
            print(f"Saved circuit statistics to {file_path}")
            
        else:
            # Original behavior: all in one file with folding
            print("Single file mode with folding is not recommended for large circuits.")
            print("Using split_files=True instead.")
            return self.visualize_qaoa_circuit(result, save_path, output_dir, 
                                            decompose_level, show_layout, show_depth, 
                                            split_files=True, max_width=max_width)
        
        print(f"\nQAOA circuit visualization completed. Files saved:")
        for file in saved_files:
            print(f"  - {file}")
        
        return saved_files


    def export_qaoa_circuit_data(self, result: Optional[TSPResult] = None,
                           format: str = 'qasm',
                           save_path: str = None,
                           output_dir: str = None) -> str:
        """
        Export QAOA circuit in various formats
        
        Parameters
        ----------
        result : TSPResult, optional
            QAOA result
        format : str
            Export format ('qasm', 'qasm3', 'qpy', 'json')
            - 'qasm': OpenQASM 2.0 (requires bound parameters)
            - 'qasm3': OpenQASM 3.0 (supports parametric circuits)
            - 'qpy': Qiskit's binary format
            - 'json': Circuit metadata
        save_path : str, optional
            Save filename (auto-generated if not specified)
        output_dir : str, optional
            Output directory
            
        Returns
        -------
        str
            Path to saved file
        """
        if result is None:
            qaoa_results = [r for r in self.results_history if r.algorithm == "qaoa"]
            if not qaoa_results:
                print("No QAOA results found.")
                return None
            result = qaoa_results[-1]
        
        # Get circuit from class instance
        if not hasattr(self, '_last_qaoa_circuit') or self._last_qaoa_circuit is None:
            print("Error: Circuit information not available in _last_qaoa_circuit")
            return None
        
        # Debug information
        print(f"Available circuits for export: {list(self._last_qaoa_circuit.keys())}")
        
        # Select circuit based on format and availability
        circuit = None
        if format in ['qasm', 'qasm3']:
            # For QASM, prefer bound circuits
            circuit = self._last_qaoa_circuit.get('bound_transpiled')
            if circuit is None:
                circuit = self._last_qaoa_circuit.get('bound_original')
            if circuit is None and format == 'qasm3':
                # QASM3 can handle parametric circuits
                circuit = self._last_qaoa_circuit.get('transpiled')
                if circuit is None:
                    circuit = self._last_qaoa_circuit.get('original')
            if circuit is None and format == 'qasm':
                # For QASM2, try to bind parameters on the fly
                param_circuit = self._last_qaoa_circuit.get('transpiled', 
                            self._last_qaoa_circuit.get('original'))
                if param_circuit and 'optimal_params' in self._last_qaoa_circuit:
                    try:
                        circuit = param_circuit.assign_parameters(self._last_qaoa_circuit['optimal_params'])
                        print("Created bound circuit on the fly for QASM export")
                    except Exception as e:
                        print(f"Error binding parameters: {e}")
        else:
            # Other formats can handle any circuit type
            circuit = (self._last_qaoa_circuit.get('bound_transpiled') or
                    self._last_qaoa_circuit.get('transpiled') or
                    self._last_qaoa_circuit.get('bound_original') or
                    self._last_qaoa_circuit.get('original'))
        
        if circuit is None:
            print("Error: No suitable circuit found")
            return None
        
        # Print circuit info
        print(f"Using circuit: {type(circuit).__name__}, "
            f"qubits: {circuit.num_qubits}, "
            f"parameters: {circuit.num_parameters}, "
            f"depth: {circuit.depth()}")
        
        # Auto-generate filename
        if save_path is None:
            
            save_path = f"qaoa_circuit.{format}"
        
        # Set output path
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            full_path = output_path / save_path
        else:
            full_path = Path(save_path)
        
        try:
            if format == 'qasm':
                # Export as OpenQASM 2.0
                if circuit.num_parameters > 0:
                    print("Warning: Circuit has unbound parameters. QASM 2.0 export may fail.")
                
                try:
                    from qiskit.qasm2 import dumps
                    qasm_str = dumps(circuit)
                except ImportError:
                    try:
                        from qiskit import qasm2
                        qasm_str = qasm2.dumps(circuit)
                    except:
                        # Fallback for older versions
                        qasm_str = circuit.qasm()
                
                with open(full_path, 'w') as f:
                    f.write(qasm_str)
                print(f"Circuit exported as OpenQASM 2.0 to {full_path}")
                
            elif format == 'qasm3':
                # Export as OpenQASM 3.0 (supports parametric circuits)
                try:
                    from qiskit.qasm3 import dumps
                    qasm3_str = dumps(circuit)
                    with open(full_path, 'w') as f:
                        f.write(qasm3_str)
                    print(f"Circuit exported as OpenQASM 3.0 to {full_path}")
                except ImportError:
                    print("Error: OpenQASM 3.0 export requires qiskit-qasm3-import package")
                    print("Install with: pip install qiskit-qasm3-import")
                    return None
                
            elif format == 'qpy':
                # Export as QPY
                try:
                    from qiskit.qpy import dump
                    with open(full_path, 'wb') as f:
                        dump(circuit, f)
                except ImportError:
                    from qiskit import qpy
                    with open(full_path, 'wb') as f:
                        qpy.dump(circuit, f)
                print(f"Circuit exported as QPY to {full_path}")
                
            elif format == 'json':
                # Export circuit metadata
                circuit_data = {
                    'num_qubits': circuit.num_qubits,
                    'depth': circuit.depth(),
                    'size': circuit.size(),
                    'num_parameters': circuit.num_parameters,
                    'gate_counts': dict(circuit.count_ops()),
                    'algorithm': 'QAOA',
                    'p_layers': result.additional_info.get('p', 'N/A'),
                    'problem_size': self.n_cities,
                    'has_parameters': circuit.num_parameters > 0,
                    'circuit_type': type(circuit).__name__
                }
                
                # Add gate details
                gate_list = []
                for instruction in circuit.data:
                    gate_info = {
                        'name': instruction.operation.name,
                        'qubits': [q._index for q in instruction.qubits],
                        'params': [float(p) if hasattr(p, '__float__') else str(p) 
                                for p in instruction.operation.params]
                    }
                    gate_list.append(gate_info)
                circuit_data['gates'] = gate_list[:20]  # First 20 gates only
                
                # Add optimal parameters if available
                if 'optimal_params' in self._last_qaoa_circuit:
                    params = self._last_qaoa_circuit['optimal_params']
                    if hasattr(params, 'tolist'):
                        circuit_data['optimal_params'] = params.tolist()
                    else:
                        circuit_data['optimal_params'] = list(params)
                
                with open(full_path, 'w') as f:
                    json.dump(circuit_data, f, indent=2)
                print(f"Circuit metadata exported as JSON to {full_path}")
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return str(full_path)
            
        except Exception as e:
            print(f"Error exporting circuit: {e}")
            import traceback
            traceback.print_exc()
            
            # Provide alternative suggestion
            if format == 'qasm' and circuit.num_parameters > 0:
                print("\nSuggestion: Try format='qasm3' for parametric circuits, "
                    "or ensure the circuit has bound parameters.")
            
            return None




