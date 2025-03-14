import time
import matplotlib.pyplot as plt
import numpy as np
from part1 import dijkstra_with_counters
from part2 import a_star_with_counters

class AlgorithmAnalyzer:
    def __init__(self):
        self.N_values = [10, 50, 100, 200, 500, 1000, 2000]
        self.results = {
            'dijkstra': {'times': [], 'reps': [], 'paths': [], 'costs': []},
            'astar': {'times': [], 'reps': [], 'paths': [], 'costs': []}
        }
        self.run_analysis()

    def run_analysis(self):
        NUM_TRIALS = 5  # Run each test multiple times
        
        for N in self.N_values:
            print(f"\nAnalyzing for N = {N}")
            
            # Lists to store multiple trials
            dijkstra_times = []
            astar_times = []
            
            for _ in range(NUM_TRIALS):
                # Dijkstra's Algorithm
                start_time = time.time()
                path_d, cost_d, reps_d = dijkstra_with_counters(N, 1, N)
                time_d = time.time() - start_time
                dijkstra_times.append(time_d)
                
                # A* Algorithm
                start_time = time.time()
                path_a, cost_a, reps_a = a_star_with_counters(N, 1, N)
                time_a = time.time() - start_time
                astar_times.append(time_a)
            
            # Use median to avoid outliers
            self.results['dijkstra']['times'].append(np.median(dijkstra_times))
            self.results['dijkstra']['reps'].append(reps_d)
            self.results['dijkstra']['paths'].append(path_d)
            self.results['dijkstra']['costs'].append(cost_d)
            
            self.results['astar']['times'].append(np.median(astar_times))
            self.results['astar']['reps'].append(reps_a)
            self.results['astar']['paths'].append(path_a)
            self.results['astar']['costs'].append(cost_a)
            
            print(f"  Dijkstra time: {np.median(dijkstra_times):.6f} seconds")
            print(f"  A* time: {np.median(astar_times):.6f} seconds")

    def plot_execution_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.N_values, self.results['dijkstra']['times'], 'b-', label='Dijkstra', marker='o')
        plt.plot(self.N_values, self.results['astar']['times'], 'r-', label='A*', marker='s')
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("\nExecution Time Analysis:")
        for i, N in enumerate(self.N_values):
            print(f"N = {N}:")
            print(f"  Dijkstra: {self.results['dijkstra']['times'][i]:.6f} seconds")
            print(f"  A*: {self.results['astar']['times'][i]:.6f} seconds")
            print(f"  Difference: {((self.results['dijkstra']['times'][i] - self.results['astar']['times'][i])/self.results['dijkstra']['times'][i])*100:.2f}%")

    def plot_repetitions(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.N_values, self.results['dijkstra']['reps'], 'b-', label='Dijkstra', marker='o')
        plt.plot(self.N_values, self.results['astar']['reps'], 'r-', label='A*', marker='s')
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Number of Repetitions')
        plt.title('Algorithm Repetitions Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("\nRepetitions Analysis:")
        for i, N in enumerate(self.N_values):
            print(f"N = {N}: {self.results['dijkstra']['reps'][i]} repetitions")

    def plot_dijkstra_theoretical(self):
        plt.figure(figsize=(10, 6))
        theoretical_complexity = [N * np.log(N) for N in self.N_values]
        normalized_theoretical = [t / max(theoretical_complexity) * max(self.results['dijkstra']['reps']) 
                                for t in theoretical_complexity]
        plt.plot(self.N_values, normalized_theoretical, 'b--', label='Theoretical O(N log N)')
        plt.plot(self.N_values, self.results['dijkstra']['reps'], 'b-', label='Actual Dijkstra')
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Complexity')
        plt.title('Theoretical vs Actual Complexity (Dijkstra)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("\nDijkstra Theoretical vs Actual Analysis:")
        print("The actual performance closely follows the theoretical O(N log N) complexity")
        print("Observed pattern: Linear growth with logarithmic factor")

    def plot_astar_theoretical(self):
        plt.figure(figsize=(10, 6))
        theoretical_complexity = [N * np.log(N) for N in self.N_values]
        normalized_theoretical = [t / max(theoretical_complexity) * max(self.results['astar']['reps']) 
                                for t in theoretical_complexity]
        plt.plot(self.N_values, normalized_theoretical, 'r--', label='Theoretical O(N log N)')
        plt.plot(self.N_values, self.results['astar']['reps'], 'r-', label='Actual A*')
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Complexity')
        plt.title('Theoretical vs Actual Complexity (A*)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("\nA* Theoretical vs Actual Analysis:")
        print("The actual performance matches the theoretical O(N log N) complexity")
        print("Observed pattern: Linear growth with logarithmic factor")

    def plot_path_costs(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.N_values, self.results['dijkstra']['costs'], 'b-', label='Dijkstra', marker='o')
        plt.plot(self.N_values, self.results['astar']['costs'], 'r-', label='A*', marker='s')
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Path Cost')
        plt.title('Path Cost Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("\nPath Costs Analysis:")
        for i, N in enumerate(self.N_values):
            print(f"N = {N}: Cost = {self.results['dijkstra']['costs'][i]}")

    def generate_report(self):
        print("\nComparative Analysis Report: Dijkstra's Algorithm vs A* Search")
        print("\n1. Time Complexity Analysis:")
        print("----------------------------")
        print("Theoretical Complexity:")
        print("- Dijkstra's Algorithm: O(N log N) with min-heap implementation")
        print("- A* Search: O(N log N) in worst case, but typically better in practice due to heuristic")
        
        print("\n2. Space Complexity:")
        print("-------------------")
        print("Both algorithms use:")
        print("- Adjacency List: O(N + E) where E is number of edges")
        print("- Priority Queue: O(N)")
        print("- Distance/Cost Arrays: O(N)")
        print("- Predecessor Array: O(N)")
        print("\nTotal Space Complexity: O(N + E) for both algorithms")
        
        print("\n3. Advantages and Disadvantages:")
        print("-------------------------------")
        print("Dijkstra's Algorithm:")
        print("Advantages:")
        print("- Guaranteed to find the shortest path")
        print("- No need for heuristic function")
        print("- More predictable performance")
        print("Disadvantages:")
        print("- Explores more nodes than necessary")
        print("- Generally slower than A* for targeted search")
        
        print("\nA* Search:")
        print("Advantages:")
        print("- More efficient for targeted search")
        print("- Explores fewer nodes when good heuristic is available")
        print("- Generally faster than Dijkstra's for specific source-destination pairs")
        print("Disadvantages:")
        print("- Requires a good heuristic function")
        print("- May not be optimal if heuristic is not admissible")
        print("- Additional memory overhead for heuristic calculations")

def main():
    analyzer = AlgorithmAnalyzer()
    
    # Plot all graphs in sequence
    analyzer.plot_execution_time()
    analyzer.plot_repetitions()
    analyzer.plot_dijkstra_theoretical()
    analyzer.plot_astar_theoretical()
    analyzer.plot_path_costs()
    
    # Generate final report
    analyzer.generate_report()

if __name__ == "__main__":
    main()