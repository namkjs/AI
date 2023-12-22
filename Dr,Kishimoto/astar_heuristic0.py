import random
import heapq
import timeit


class State:
    def __init__(self, num_cities):
        self.visited = [False] * num_cities
        self.num_visited = 0
        self.current_id = 0
        self.path = []  # To store the path

    def hash_value(self):
        hash_val = self.current_id * (2 ** len(self.visited))
        for i in range(len(self.visited)):
            if self.visited[i]:
                hash_val += 2**i
        return hash_val


class StateWrapper:
    def __init__(self, cost, state):
        self.cost = cost
        self.state = state

    def __lt__(self, other):
        return self.cost < other.cost


def generate_random_tsp_instance(num_cities, seed):
    random.seed(seed)
    costs = [
        [0 if i == j else random.randint(1, 100) for j in range(num_cities)]
        for i in range(num_cities)
    ]
    return costs


def heuristic(state, costs):
    return 0


def a_star_algorithm(state, heuristic_func, costs, time_limit):
    start_time = timeit.default_timer()
    REACHED = [False] * (len(state.visited) * (2 ** len(state.visited)))
    priority_queue = [StateWrapper(0, state)]
    heapq.heapify(priority_queue)
    expanded_nodes = 0
    generated_nodes = 0

    while priority_queue:
        current_wrapper = heapq.heappop(priority_queue)
        current_cost, current_state = current_wrapper.cost, current_wrapper.state
        generated_nodes += 1

        if timeit.default_timer() - start_time > time_limit:
            print("Time limit exceeded.")
            return None, None, None, None, None

        if REACHED[current_state.hash_value()]:
            continue

        REACHED[current_state.hash_value()] = True
        expanded_nodes += 1

        if (
            current_state.num_visited == len(costs)
            and current_state.current_id == 0
            and current_state.visited[0]
        ):
            end_time = timeit.default_timer()
            run_time = end_time - start_time
            return (
                run_time,
                current_cost,
                expanded_nodes,
                generated_nodes,
                current_state.path + [current_state.current_id],  # Return the path
            )

        for next_city in range(len(costs)):
            if not current_state.visited[next_city]:
                new_state = State(len(costs))
                new_state.visited = current_state.visited.copy()
                new_state.visited[next_city] = True
                new_state.num_visited = current_state.num_visited + 1
                new_state.current_id = next_city
                new_state.path = current_state.path + [
                    current_state.current_id
                ]  # Update the path

                new_cost = current_cost + costs[current_state.current_id][next_city]
                priority = new_cost + heuristic_func(new_state, costs)

                heapq.heappush(priority_queue, StateWrapper(priority, new_state))

    print("No solution found within the time limit.")
    return None, None, None, None, None


# Run experiments for multiple instances with the same seed for each N
num_cities_list = [5, 10, 11, 12]
num_instances = 5
time_limit = 20 * 60  # 20 minutes in seconds

for num_cities in num_cities_list:
    for instance in range(1, num_instances + 1):
        seed = instance
        costs = generate_random_tsp_instance(num_cities, seed)
        initial_state = State(num_cities)

        (
            run_time,
            optimal_cost,
            expanded_nodes,
            generated_nodes,
            optimal_path,
        ) = a_star_algorithm(initial_state, heuristic, costs, time_limit)

        if run_time is not None and optimal_cost is not None:
            print(f"Results for {num_cities} cities with Seed {seed} are:")
            print(f"Run Time is: {run_time:.6f} seconds")
            print(f"Optimal Path Cost is: {optimal_cost:.2f}")
            print("Optimal Path:", optimal_path)  # Print the optimal path
            print(f"The number of Expanded Nodes are: {expanded_nodes}")
            print(f"The number of Generated Nodes are: {generated_nodes}")
            print("************************")