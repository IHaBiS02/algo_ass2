from tsp import TSP
import time
from tqdm import trange, tqdm
import numpy as np
import cupy as cp
from itertools import combinations
from calculate_func import calculate_distance_4value

def calculate_distance(i, j):
    # return math.sqrt((i[0]-j[0])** 2 + (i[1]-j[1])**2)
    return calculate_distance_4value(i[0], i[1], j[0], j[1])


def held_karp(nodes):
    n = len(nodes)
    if n == 0:
        return [], 0

    dist_matrix = np.zeros((n, n))
    for i in trange(n):
        for j in range(i, n):
            dist = calculate_distance(nodes[i], nodes[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    C = {}

    for k in range(1, n):
        C[(1 | 1 << k, k)] = (dist_matrix[0, k], 0)

    for subset_size in trange(3, n + 1):
        for subset_tuple in combinations(range(1, n), subset_size - 1):
            subset_mask = 1
            for node in subset_tuple:
                subset_mask |= (1 << node)

            for j in subset_tuple:
                prev_mask = subset_mask & ~(1 << j)
                min_cost = float('inf')
                best_prev_node = -1

                for k in subset_tuple:
                    if k == j:
                        continue
                    
                    cost, _ = C.get((prev_mask, k), (float('inf'), -1))
                    new_cost = cost + dist_matrix[k, j]

                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_prev_node = k
                
                if best_prev_node != -1:
                    C[(subset_mask, j)] = (min_cost, best_prev_node)
    
    full_mask = (1 << n) - 1
    min_tour_cost = float('inf')
    last_node = -1

    for j in trange(1, n):
        cost, _ = C.get((full_mask, j), (float('inf'), -1))
        final_cost = cost + dist_matrix[j, 0]
        if final_cost < min_tour_cost:
            min_tour_cost = final_cost
            last_node = j

    tour = []
    current_mask = full_mask
    current_node = last_node

    while current_node != 0:
        tour.append(current_node)
        _, prev_node = C[(current_mask, current_node)]
        current_mask &= ~(1 << current_node)
        current_node = prev_node
    
    tour.append(0)
    tour.reverse()

    return tour, min_tour_cost

def held_karp_optimized(nodes):
    n = len(nodes)
    if n == 0:
        return [], 0

    nodes_arr = np.array(nodes)
    diffs = nodes_arr[:, np.newaxis, :] - nodes_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diffs**2, axis=-1))

    C = np.full((1 << n, n), np.inf)
    P = np.zeros((1 << n, n), dtype=int)

    for k in range(1, n):
        mask = 1 | (1 << k)
        C[mask, k] = dist_matrix[0, k]

    for subset_size in trange(3, n + 1):
        for subset_tuple in combinations(range(1, n), subset_size - 1):
            subset_mask = 1 | sum(1 << node for node in subset_tuple)
            
            for j in subset_tuple:
                prev_mask = subset_mask & ~(1 << j)
                
                k_indices = np.array([k for k in subset_tuple if k != j])
                
                if k_indices.size == 0:
                    continue

                costs = C[prev_mask, k_indices] + dist_matrix[k_indices, j]
                
                min_idx = np.argmin(costs)
                min_cost = costs[min_idx]
                
                best_prev_node = k_indices[min_idx]

                if min_cost != np.inf:
                    C[subset_mask, j] = min_cost
                    P[subset_mask, j] = best_prev_node
    
    full_mask = (1 << n) - 1
    final_nodes = np.arange(1, n)
    
    final_costs = C[full_mask, final_nodes] + dist_matrix[final_nodes, 0]
    
    min_tour_cost = np.min(final_costs)
    last_node = final_nodes[np.argmin(final_costs)]

    tour = []
    current_mask = full_mask
    current_node = last_node

    if min_tour_cost == np.inf:
        print("Error: Optimal tour could not be found.")
        return [], float('inf')

    while current_node != 0:
        tour.append(current_node)
        prev_node = P[current_mask, current_node]
        current_mask &= ~(1 << current_node)
        current_node = prev_node
    
    tour.append(0)
    tour.reverse()

    return tour, min_tour_cost

if __name__ == "__main__":
    print("Starting TSP solution with Held-Karp algorithm.")
    
    print("load tsp")
    tsp = TSP('a280.tsp')
    print("start processing")
    

    st_tsp = time.time()
    tsp_tour, tsp_cost = held_karp_optimized(tsp.node_coord_section[:25])
    et_tsp = time.time()
    
    print(f"\nHeld-Karp TSP finished, elapsed_time: {et_tsp - st_tsp:.4f} seconds")
    print("\n" + "="*50 + "\n")

    print("--- Final TSP Tour Results ---")
    print(f"Total Tour Cost: {tsp_cost:.4f}")
    
    print("Optimal Tour Path:")
    print(" -> ".join(map(str, tsp_tour)))
