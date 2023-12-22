import heapq
import random
import time

class State:
    def __init__(self, N):
        self.visited = [False] * N
        self.num_visited = 0
        self.current_id = 0
        self.path = []
        self.cost = 0

    def __lt__(self, other):
        return False

def min_out_heuristic(costs, state):
    unvisited_cities = [costs[state.current_id][j] for j in range(len(costs)) if not state.visited[j]]
    return min(unvisited_cities) if unvisited_cities else 0

def a_star(N, costs):
    start_state = State(N)
    start_state.visited[0] = True
    start_state.num_visited = 1
    frontier = [(0, start_state)]
    reached = [None] * (N * 2 ** N)
    expanded_nodes = 0
    generated_nodes = 0

    while frontier:
        _, current_state = heapq.heappop(frontier)
        expanded_nodes += 1

        if current_state.num_visited == N:
            current_state.cost += costs[current_state.current_id][0]
            current_state.path.append(0)
            return current_state.cost, current_state.path, expanded_nodes, generated_nodes

        for next_city in range(1, N):
            if not current_state.visited[next_city]:
                generated_nodes += 1
                next_state = State(N)
                next_state.visited = current_state.visited.copy()
                next_state.visited[next_city] = True
                next_state.num_visited = current_state.num_visited + 1
                next_state.current_id = next_city
                next_state.path = current_state.path + [next_city]
                next_state.cost = current_state.cost + costs[current_state.current_id][next_city]
                next_cost = next_state.cost + min_out_heuristic(costs, next_state)

                S = next_state.current_id * 2 ** N + sum(2 ** i if visited else 0 for i, visited in
                                                         enumerate(next_state.visited))
                if reached[S] is None or next_state.cost < reached[S]:
                    reached[S] = next_state.cost
                    heapq.heappush(frontier, (next_cost, next_state))

def calculate_average(results, denominator):
    solved_problems = len(results)
    if solved_problems == 0:
        return "NA"
    
    average_runtime = sum(result["time"] for result in results) / solved_problems
    average_cost = sum(result["result"] for result in results) / solved_problems
    average_expanded_nodes = sum(result["expanded_nodes"] for result in results) / solved_problems
    average_generated_nodes = sum(result["generated_nodes"] for result in results) / solved_problems

    return solved_problems, average_runtime, average_cost, average_expanded_nodes, average_generated_nodes

def main():
    N_values = [5, 10, 11, 12]
    seeds = [1, 2, 3, 4, 5]
    results = []

    for N in N_values:
        for seed in seeds:
            random.seed(seed)
            costs = [[random.randint(1, 100) for _ in range(N)] for _ in range(N)]
            start_time = time.time()
            result, optimal_path, expanded_nodes, generated_nodes = a_star(N, costs)
            end_time = time.time()

            results.append({
                "N": N,
                "seed": seed,
                "result": result,
                "time": end_time - start_time,
                "expanded_nodes": expanded_nodes,
                "generated_nodes": generated_nodes,
                "optimal_path": optimal_path
            })

    # Prepare and print the table
    print("N\tSeed\tSolved\tAvg Time\tAvg Cost\tAvg Expanded Nodes\tAvg Generated Nodes\tOptimal Path")
    for N in N_values:
        for seed in seeds:
            subset = [result for result in results if result["N"] == N and result["seed"] == seed]
            solved, avg_time, avg_cost, avg_expanded_nodes, avg_generated_nodes = calculate_average(subset, 5)
            optimal_path_str = " ".join(map(str, subset[0]["optimal_path"])) if solved > 0 else "NA"
            print(f"{N}\t{seed}\t{solved}\t{avg_time:.4f}\t{avg_cost:.2f}\t{avg_expanded_nodes:.2f}\t{avg_generated_nodes:.2f}\t{optimal_path_str}")

if __name__ == "__main__":
    main()