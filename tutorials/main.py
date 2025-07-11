import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime






from tsp_benchmark import *

# Usage example and test (Qiskit v2.0 compatible)
if __name__ == "__main__":
    print("=== TSP Benchmark Library (Qiskit v2.0 Compatible) ===\n")

    # Set output directory
    output_base_dir = "tsp_benchmark_results"
    timestamp = datetime.now().strftime('%Y%m%d')
    output_dir = f"{output_base_dir}/{timestamp}"

    print(f"Output directory: {output_dir}\n")

    # TSP problem using distance matrix only
    print("TSP problem using distance matrix only (Qiskit v2.0 compatible)")

    # Small distance matrix for quick test
    
    distance_matrix = np.array([
        [0.0, 2.5, 3.2],
        [2.5, 0.0, 1.9],
        [3.2, 1.9, 0.0]
    ])

    '''
    distance_matrix = np.array([
    [0.0, 2.5, 3.2, 1.8],
    [2.5, 0.0, 1.9, 3.7],
    [3.2, 1.9, 0.0, 2.8],
    [1.8, 3.7, 2.8, 0.0],
    ])
    distance_matrix = np.array([
    [0.0, 2.5, 3.2, 1.8, 4.1],
    [2.5, 0.0, 1.9, 3.7, 2.2],
    [3.2, 1.9, 0.0, 2.8, 3.5],
    [1.8, 3.7, 2.8, 0.0, 2.9],
    [4.1, 2.2, 3.5, 2.9, 0.0]
    ])
    '''

    print(f"Distance matrix shape: {distance_matrix.shape}")
    print("Distance matrix:")
    print(distance_matrix)

    # Create TSP instance
    tsp = TSPBenchmark(distance_matrix=distance_matrix)

    # Check estimated coordinates
    if tsp.coordinates:
        print("\nEstimated coordinates:")
        for i, (x, y) in enumerate(tsp.coordinates):
            print(f"City {i}: ({x:.5f}, {y:.5f})")

    # Visualize city layout (save to graphs directory)
    tsp.visualize_cities("TSP Cities Layout (Qiskit v2.0)",
                        save_path="cities_layout.png",
                        output_dir=f"{output_dir}/graphs")

    # Execute algorithms
    print("\n=== Algorithm Execution (Qiskit v2.0 compatible) ===")

    # 1. Held-Karp algorithm (exact solution)
    print("\n1. Held-Karp algorithm (exact solution)")
    result_hk = tsp.solve(algorithm=Algorithm.HELD_KARP)
    print(f"Result: {result_hk}")

    # 2. Simulated Annealing
    print("\n2. Simulated Annealing")
    result_sa = tsp.solve(
        algorithm=Algorithm.SIMULATED_ANNEALING,
        num_reads=100,
        uniform_penalty_weight=100.0
    )
    print(f"Result: {result_sa}")

    # 3. QAOA execution
    print("\n3. QAOA  - Qiskit v2.0 compatible")
    result_qaoa = tsp.solve(
        algorithm=Algorithm.QAOA,
        p=3,
        optimizer='COBYLA',
        maxiter=500,
        uniform_penalty_weight=100.0,
        shots=4096,
        backend_type='auto',
        use_gpu=True
    )
    print(f"Result: {result_qaoa}")

    # Visualize results (save to graphs directory)
    print("\n=== Result Visualization (Qiskit v2.0 compatible) ===")
    graphs_dir = f"{output_dir}/graphs"

    tsp.visualize_solution(result_hk, "Held-Karp Solution (v2.0)",
                          save_path="hk_solution.png",
                          output_dir=graphs_dir)
    tsp.visualize_solution(result_sa, "SA Solution (v2.0)",
                          save_path="sa_solution.png",
                          output_dir=graphs_dir)
    tsp.visualize_solution(result_qaoa, "QAOA Solution (v2.0)",
                          save_path="qaoa_solution.png",
                          output_dir=graphs_dir)

    # Display algorithm comparison summary
    print("\n=== Algorithm Comparison Summary ===")
    tsp.display_comparison_summary()

    # Export comparison results to files (save to tables directory)
    print("\n=== Export Comparison Results ===")
    tables_dir = f"{output_dir}/tables"
    output_files = tsp.export_comparison_table(
        output_format='all',
        prefix='tsp_benchmark_results',
        output_dir=tables_dir
    )
    print(f"Output files: {output_files}")

    # Algorithm comparison
    print("\n=== Algorithm Comparison (Qiskit v2.0 compatible) ===")
    comparison_df = tsp.compare_algorithms()
    print(comparison_df.to_string())

    # Generate comparison chart (save to graphs directory)
    tsp.plot_comparison(save_path="algorithm_comparison.png",
                       output_dir=graphs_dir)

    # QAOA convergence chart (save to graphs directory)
    tsp.plot_qaoa_convergence(result_qaoa,
                             save_path="qaoa_convergence.png",
                             output_dir=graphs_dir)

    # Generate report (save to reports directory)
    reports_dir = f"{output_dir}/reports"
    tsp.generate_report(save_path="tsp_benchmark_report.txt",
                       output_dir=reports_dir)
    
    # save circuit
    circuit_dir = f"{output_dir}/circuit"
    tsp.visualize_qaoa_circuit(result_qaoa, 
                          save_path="qaoa_circuit_visualization.png",
                          output_dir=circuit_dir)

    # export circuit using qasm style
    tsp.export_qaoa_circuit_data(result_qaoa, 
                                format='qasm',
                                output_dir=circuit_dir)

    # export circuit using json style
    tsp.export_qaoa_circuit_data(result_qaoa, 
                                format='json',
                                output_dir=circuit_dir)

    print("\n=== Qiskit v2.0 Compatible Benchmark Complete ===")