import heapq

def dijkstra_with_counters(N, S, D):
    """
    Runs Dijkstra's algorithm with repetition counters.
    N: number of nodes (1..N)
    S: source node
    D: destination node
    Returns: (shortest_path, cost_of_path, total_repetitions)
    """
    
    # ----------------------------
    # Build adjacency list
    # ----------------------------
    # adjacency[node] = list of (neighbor, weight)
    adjacency = [[] for _ in range(N+1)]
    
    # Counter for adjacency building
    adjacency_build_count = 0  # Cost = 1 per iteration in the loops
    
    for i in range(1, N+1):
        # For j in [i-3, i+3], skipping i, but within 1..N
        for j in range(max(1, i-3), min(N, i+3)+1):
            adjacency_build_count += 1  # increment for each check
            if j != i:
                w = i + j  # weight
                adjacency[i].append((j, w))
    
    # ----------------------------
    # Initialize distances and heap
    # ----------------------------
    dist = [float('inf')] * (N+1)  # dist[i] = current best distance from S to i
    dist[S] = 0
    
    # predecessor[i] = node used to reach i with minimal dist
    predecessor = [-1] * (N+1)
    
    # build a list of (distance, node) pairs
    min_heap = [(0, S)]  # Python's heapq is a min-heap
    
    visited = [False] * (N+1)
    
    # ----------------------------
    # Counters
    # ----------------------------
    extract_min_count = 0      # how many times we pop from heap
    relaxation_count = 0       # how many times we do relaxation checks
    decrease_key_count = 0     # how many times we do the "dist[v]" update
    
    # ----------------------------
    # Dijkstra's main loop
    # ----------------------------
    while min_heap:
        extract_min_count += 1  # we count each pop
        current_dist, u = heapq.heappop(min_heap)
        
        if visited[u]:
            continue
        visited[u] = True
        
        # If we've reached D, we can stop if desired
        if u == D:
            break
        
        # Relax edges
        for (v, w_uv) in adjacency[u]:
            relaxation_count += 1
            if not visited[v]:
                alt = dist[u] + w_uv
                if alt < dist[v]:
                    dist[v] = alt
                    predecessor[v] = u
                    decrease_key_count += 1
                    # push new distance
                    heapq.heappush(min_heap, (dist[v], v))
    
    # ----------------------------
    # Reconstruct the path from S to D
    # ----------------------------
    if dist[D] == float('inf'):
        # no path
        return ([], float('inf'), adjacency_build_count + extract_min_count 
                + relaxation_count + decrease_key_count)
    
    # build the path in reverse
    path = []
    node = D
    while node != -1:
        path.append(node)
        node = predecessor[node]
    
    path.reverse()  # since we built it backwards
    
    total_repetitions = adjacency_build_count + extract_min_count + relaxation_count + decrease_key_count
    
    return (path, dist[D], total_repetitions)


if __name__ == "__main__":
    # Example usage for N=10, S=1, D=10
    # You can change these or prompt the user
    N = int(input("Enter N (<20): "))
    S = int(input("Enter S: "))
    D = int(input("Enter D: "))
    
    shortest_path, cost, total_reps = dijkstra_with_counters(N, S, D)
    
    if cost == float('inf'):
        print(f"No path exists from {S} to {D}.")
    else:
        print(f"Shortest path from {S} to {D} is: {shortest_path}")
        print(f"Total cost of this path is: {cost}")
    print(f"Total repetition count: {total_reps}")
