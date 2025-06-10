from calculate_func import calculate_distance_4value
from mst import create_mst, create_mst_gpu, create_mst_gpu_prim, create_mst_cpu_prim
from tsp import TSP
import time
from tqdm import trange
import numpy as np
import random
def calculate_distance(i, j):
    # return math.sqrt((i[0]-j[0])** 2 + (i[1]-j[1])**2)
    return calculate_distance_4value(i[0], i[1], j[0], j[1])

def _mst_2_approximation_tsp(nodes, mst_edges, start_node=0):
    num_nodes = len(nodes)
    # print(num_nodes)
    if not mst_edges: return [], 0
    
    adj_list = {i: [] for i in range(num_nodes)}
    for u, v, _ in mst_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    # print(adj_list)
    tour_order = []
    visited = [False for _ in range(num_nodes)]
    stack = [start_node] 

    while stack:
        u = stack.pop()
        if not visited[u]:
            visited[u] = True
            tour_order.append(u)
            for v in reversed(adj_list[u]):
                if not visited[v]:
                    stack.append(v)

    total_tour_cost = 0
    for i in range(len(tour_order) - 1):
        u, v = tour_order[i], tour_order[i+1]
        total_tour_cost += calculate_distance(nodes[u], nodes[v])
    
    if tour_order:
        last_node, first_node = tour_order[-1], tour_order[0]
        total_tour_cost += calculate_distance(nodes[last_node], nodes[first_node])

    return tour_order, total_tour_cost



def mst_2_approximation_tsp(nodes):
    mst = create_mst_gpu_prim(nodes)
    tsp_tour, tsp_cost = _mst_2_approximation_tsp(nodes, mst)


def mst_2_approximation_tsp_novelty(nodes):
    mst = create_mst_gpu_prim(nodes)
    best_tour_of_all = None
    best_cost_of_all = float("inf")
    for i in np.random.choice(range(len(nodes)), min(200, len(nodes)), replace=False):
        tsp_tour, tsp_cost = _mst_2_approximation_tsp(nodes, mst, i)
        if tsp_cost < best_cost_of_all:
            best_tour_of_all = tsp_tour
            best_cost_of_all = tsp_cost
    return best_tour_of_all, best_cost_of_all


if __name__ == "__main__":
    print("load tsp")
    tsp = TSP('a280.tsp')
    print("start processing")
    # print(len(tsp.node_coord_section))
    st = time.time()
    # print(tsp.node_coord_section[:10], len(tsp.node_coord_section))
    used_nodes = tsp.node_coord_section[:25]
    mst = create_mst_gpu_prim(used_nodes)
    et = time.time()
    print(f"processing finished, elasped_time : {et-st}")
    print("\n" + "="*50 + "\n")
    # print(mst[:10])

    print("### Step 2: Applying MST-based 2-Approximation Algorithm ###")
    st_tsp = time.time()
    tsp_tour, tsp_cost = _mst_2_approximation_tsp(used_nodes, mst)
    et_tsp = time.time()
    print(f"2-Approximation TSP finished, elapsed_time: {et_tsp - st_tsp:.4f} seconds")

    print("\n--- Final TSP Tour Results ---")
    print(f"Total Tour Cost: {tsp_cost:.4f}")
    print("Tour Path (first 20 nodes):")

    print(" -> ".join(map(str, tsp_tour[:20])))
