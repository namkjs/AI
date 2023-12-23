from astar_heuristic0 import *
from IDA_heuristic0 import algorithm
from IDA_min_out_heuristic import algorithm as algorithm1
import random
import timeit

# ... (Các hàm và lớp cần thiết)


def measure_algorithm_time(algorithm_func, *args):
    start_time = timeit.default_timer()
    algorithm_func(*args)
    end_time = timeit.default_timer()
    return end_time - start_time


def compare_algorithm_runtimes(num_cities_list, num_instances, time_limit):
    algorithms = [
        ("A*", a_star_algorithm),
        ("Recursive Algorithm", algorithm),
        ("IDA*", algorithm1),
    ]

    for num_cities in num_cities_list:
        for instance in range(1, num_instances + 1):
            seed = instance
            costs = generate_random_tsp_instance(num_cities, seed)
            initial_state = State(num_cities)

            print(f"Comparing runtimes for {num_cities} cities with Seed {seed}:")
            for alg_name, alg_func in algorithms:
                time_taken = measure_algorithm_time(
                    alg_func, initial_state, heuristic, costs, time_limit
                )
                print(f"{alg_name} - Time taken: {time_taken:.6f} seconds")
            print("---------------------------------------------------------------")


if __name__ == "__main__":
    num_cities_list = [5, 10, 11, 12]
    num_instances = 5
    time_limit = 20 * 60  # 20 minutes in seconds

    compare_algorithm_runtimes(num_cities_list, num_instances, time_limit)