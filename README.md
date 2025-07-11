# TSP Benchmark Library

A comprehensive benchmarking library for comparing classical and quantum algorithms on the Traveling Salesman Problem (TSP).

## Features

- **Multiple Algorithms**: Compare Simulated Annealing, QAOA (Quantum Approximate Optimization Algorithm), and exact solutions (Held-Karp)
- **Unified Interface**: Simple API for running different algorithms with the same problem instance
- **Comprehensive Evaluation**: Built-in metrics for solution quality, feasibility, and performance
- **Visualization Tools**: Plot cities, solutions, and algorithm comparisons
- **Export Capabilities**: Save results in multiple formats (CSV, Excel, HTML, JSON)

## Installation

```bash
pip install git+https://github.com/thedaemon-wizard/tsp_benckmark
```

### Dependencies

- Python 3.8+
- NumPy, Matplotlib, Pandas, SciPy
- OMMX, ommx-openjij-adapter, OpenJij
- Qiskit, qiskit-aer
- scikit-learn (optional, for coordinate estimation)

## Quick Start

### Google Colab

```python
# Install the library
!pip install git+https://github.com/thedaemon-wizard/tsp_benckmark

# Import and use
from tsp_benchmark import TSPBenchmark, Algorithm, TSPResult

# Create a TSP instance with coordinates
coordinates = [
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0)
]
tsp = TSPBenchmark(coordinates=coordinates)

# Solve with different algorithms
result_sa = tsp.solve(algorithm=Algorithm.SIMULATED_ANNEALING)
result_qaoa = tsp.solve(algorithm=Algorithm.QAOA, p=2)
result_exact = tsp.solve(algorithm=Algorithm.HELD_KARP)

# Compare results
tsp.display_comparison_summary()
```

## Available Algorithms

### 1. Simulated Annealing
- Implementation via OMMX-OpenJij adapter
- Parameters: `num_reads`, `uniform_penalty_weight`

### 2. QAOA (Quantum Approximate Optimization Algorithm)
- Quantum-inspired optimization using Qiskit
- Parameters: `p` (layers), `optimizer`, `shots`, `maxiter`

### 3. Held-Karp (Exact Solution)
- Dynamic programming approach
- Suitable for small problems (n ≤ 20)

## Basic Usage

### Creating a TSP Instance

```python
# From coordinates
tsp = TSPBenchmark(coordinates=[(x1, y1), (x2, y2), ...])

# From distance matrix
distance_matrix = np.array([[0, d01, d02], [d10, 0, d12], [d20, d21, 0]])
tsp = TSPBenchmark(distance_matrix=distance_matrix)
```

### Solving the Problem

```python
# Simulated Annealing
result = tsp.solve(
    algorithm=Algorithm.SIMULATED_ANNEALING,
    num_reads=100,
    uniform_penalty_weight=20.0
)

# QAOA
result = tsp.solve(
    algorithm=Algorithm.QAOA,
    p=2,                    # QAOA layers
    optimizer='COBYLA',     # Classical optimizer
    shots=2048             # Measurement shots
)

# Exact solution
result = tsp.solve(algorithm=Algorithm.HELD_KARP)
```

### Analyzing Results

```python
# Display comparison summary
tsp.display_comparison_summary()

# Export results
output_files = tsp.export_comparison_table(
    output_format='all',  # 'csv', 'excel', 'html', 'json', or 'all'
    prefix='tsp_results'
)

# Visualize solutions
tsp.visualize_solution(result, "Algorithm Solution")
tsp.plot_comparison()
```

## Result Object

Each algorithm returns a `TSPResult` object containing:
- `path`: The visiting order of cities
- `distance`: Total travel distance
- `feasible`: Whether the solution satisfies all constraints
- `computation_time`: Execution time in seconds
- `additional_info`: Algorithm-specific information

## Visualization

```python
# Visualize city locations
tsp.visualize_cities("City Layout")

# Visualize a specific solution
tsp.visualize_solution(result, "Solution Path")

# Plot algorithm comparison
tsp.plot_comparison()

# QAOA convergence (if applicable)
tsp.plot_qaoa_convergence(result_qaoa)
```

## Example Notebook

For a comprehensive tutorial, see the [example notebook](examples/tsp_benchmark_tutorial.ipynb).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using [OMMX](https://github.com/Jij-Inc/ommx) for mathematical modeling
- Quantum algorithms implemented with [Qiskit](https://qiskit.org/)
- Classical optimization via [OpenJij](https://github.com/OpenJij/OpenJij)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{tsp_benchmark,
  title = {TSP Benchmark Library},
  author = {Amon koike},
  year = {2025},
  url = {https://github.com/thedaemon-wizard/tsp_benckmark}
}
```