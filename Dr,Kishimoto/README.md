# Traveling Salesperson Problem Solver

## Project Overview

In this team project, consisting of 4-5 members, the goal is to implement search algorithms to calculate optimal solution costs for the Traveling Salesperson Problem (TSP), a well-known NP-hard problem. The performance of these algorithms will be compared.

## The TSP Problem

In TSP, given N cities, starting with the first city, a salesperson must visit each city exactly once and finally return to the first city. The objective is to travel with the smallest sum of edge costs on the route.

For more details, refer to [Wikipedia](https://en.wikipedia.org/wiki/Travelling_salesman_problem).

## Implemented Algorithms

1. **IDA* with heuristic function h(n)=0**
2. **IDA* with the min-out heuristic function**
3. **A* with h(n)=0 (Dijkstra’s algorithm)**
4. **A* with the min-out heuristic function**

## Specifications

- Programming Language: Any
- Number of Cities (N): 5, 10, 11, 12
- Five problems for each N for performance evaluation
- Cost(i, j): Integer in range [1, 100]
- Pseudo-random generator for cost initialization
- Seeds: 1, 2, 3, 4, 5 for consistent problem solving
- Time limit: 20 minutes per problem

## Performance Metrics

Prepare a table in the project report summarizing the performance for each algorithm and for each N:

1. Number of solved problems
2. Average run time
3. Average optimal path cost
4. Average number of expanded nodes
5. Average number of generated nodes

- Use the number of solved problems as a denominator for average calculations.
- Truncate values as needed.
- Write "NA" if an algorithm does not solve any problems.
- Assume 20 minutes runtime for unsolved problems in average runtime calculation.

## State Representation

```cpp
class State:
  bool visited[N]; // Flags for visited cities (true if visited, true if coming back for id=0)
  int num_visited; // Number of visited cities
  int current_id; // Id of the current city

## Implementation Details

### Cycle Check

With the provided implementation, the search space inherently avoids creating cycles. The states where the salesperson is at city 0 at the beginning and at the end are distinct, eliminating the need for a cycle check. Initial visited flags are false at city 0, and when returning, all visited flags become true.

### A* Priority Queue

For A*, a priority queue is essential. You can utilize existing libraries such as:

- [Python heapq](https://docs.python.org/3/library/heapq.html)
- [C++ priority_queue](https://en.cppreference.com/w/cpp/container/priority_queue)

### REACHED Structure for A*

In the case of A*, the REACHED structure is crucial. Implement it as follows:

- Let N be the number of cities.
- Define hash value S as S = ID × 2^N + a_0 + a_1 + … + a_{N-1}.
- For each visited[i], if false, set a_i = 0; if true, set a_i = 2^i.
- ID represents the current city's id.

### Speed Consideration

Considering speed differences in various programming languages, no points will be deducted for slow run times stemming from language performance. However, a faster program in real-time is advantageous.

### Encapsulation with Classes

While not mandatory, using classes (e.g., in C++/Python) enhances code readability. Encapsulation with classes provides a clearer structure for the implementation.

### Identical Optimal Solutions

As the goal is to find optimal solutions, ensure that the optimal solution costs are identical when the same problem is solved by different algorithms. Any differences may indicate a bug in the implementation.

### Handling Tied Optimal Scores

Be aware that tied optimal scores may exist, leading to different optimal routes. Account for this variation in your analyses.

### Function/Method Documentation

Provide brief explanations for each function/method, including their arguments, returned values, and functionality. Incorporate these comments directly into your source code.

## Advanced Topic - Enhanced IDA*

In response to the drawback of IDA* reexpanding the same nodes, an advanced enhancement involves implementing a hash table (transposition table). Refer to [this link](https://webdocs.cs.ualberta.ca/~tony/TechnicalReports/tr-ri-93-120.pdf) for details.

Assume permission to preserve all states in the hash table, similar to A* (i.e., REACHED). Implement this enhanced version of IDA* to solve the TSP with h(n)=0 and the min-out heuristic. Compare its performance with the other algorithms.

## Submission Details

- **Due Date:** 11:59 PM December 22nd (sharp deadline).
- **Submission Location:** Dr. Kien will receive the files; follow his guidelines.
- **Submission Format:** Submit one zipped file containing the source code and a PDF file of the performance comparison tables.

## Important Notes

- Team members will receive the same marks; work effectively as a team.
- Even if incomplete, submit the source code and tables showing your progress by the due date.
- Begin early to allow ample time for implementation.
- Consult with the instructor promptly for guidance on difficulties; hints and supplementary information will be provided as needed.
