from calculate_func import calculate_distance_4value
import cupy as cp
from tsp import TSP
import time
import math
from tqdm import trange, tqdm
import numpy as np

def calculate_distance(i, j):
    return calculate_distance_4value(i[0], i[1], j[0], j[1])

def find_parent(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find_parent(parent, parent[i])
    return parent[i]

def union_parent(parent, rank, x, y):
    root_x = find_parent(parent, x)
    root_y = find_parent(parent, y)

    if root_x != root_y:
        if rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_x] = root_y
            if rank[root_x] == rank[root_y]:
                rank[root_y] += 1
        return True
    return False

def create_mst(nodes):
    num_nodes = len(nodes)
    edges = []

    for i in trange(num_nodes):
        for j in range(i+1, num_nodes):
            distance = calculate_distance(nodes[i], nodes[j])
            edges.append((distance, i, j))
    
    edges.sort()

    parent = list(range(num_nodes))
    rank = [0 for _ in range(num_nodes)]

    mst_edges = []
    mst_cost = 0

    for edge in tqdm(edges):
        distance, u, v = edge
        if union_parent(parent, rank, u, v):
            mst_edges.append((u, v, distance))
            mst_cost += distance
            if len(mst_edges) == num_nodes - 1:
                break
    
    print(f"Total MST Cost: {mst_cost}")
    return mst_edges

def create_mst_gpu(nodes):

    num_nodes = len(nodes)
 
    nodes_gpu = cp.asarray(nodes, dtype=cp.float32)
    
    dist_matrix_sq = cp.sum((nodes_gpu[:, cp.newaxis, :] - nodes_gpu[cp.newaxis, :, :]) ** 2, axis=-1)
    
    i_indices, j_indices = cp.triu_indices(num_nodes, k=1)
    
    distances = cp.sqrt(dist_matrix_sq[i_indices, j_indices])

    edges_gpu = cp.stack((distances, i_indices.astype(cp.float32), j_indices.astype(cp.float32)), axis=1)

    sorted_edges_gpu = edges_gpu[cp.argsort(edges_gpu[:, 0])]


    edges = np.asarray(sorted_edges_gpu.get())

    parent = list(range(num_nodes))
    rank = [0] * num_nodes

    mst_edges = []
    mst_cost = 0

    for edge in tqdm(edges):
        distance, u, v = edge

        u, v = int(round(u)), int(round(v))

        if union_parent(parent, rank, u, v):
            mst_edges.append((u, v, distance))
            mst_cost += distance

            if len(mst_edges) == num_nodes - 1:
                break
    
    return mst_edges  

def create_mst_gpu_prim(nodes):
    num_nodes = len(nodes)
    if num_nodes == 0:
        return []
    
    print("Moving data to GPU and initializing for Prim's algorithm...")
    nodes_gpu = cp.asarray(nodes, dtype=cp.float32)

    dist = cp.full(num_nodes, cp.inf, dtype=cp.float32)

    parent = cp.full(num_nodes, -1, dtype=cp.int32)

    visited = cp.zeros(num_nodes, dtype=cp.bool_)

    dist[0] = 0
    
    mst_edges = []
    mst_cost = 0

    for _ in trange(num_nodes, desc="Building MST with Prim's"):
        temp_dist = cp.where(visited, cp.inf, dist)
        u = int(cp.argmin(temp_dist))

        if cp.isinf(dist[u]):
            break

        visited[u] = True
        
        if parent[u] != -1:
            p = int(parent[u])
            d = float(dist[u])
            mst_edges.append((p, u, d))
            mst_cost += d

        distances_from_u = cp.sqrt(cp.sum((nodes_gpu - nodes_gpu[u])**2, axis=1))

        update_mask = (~visited) & (distances_from_u < dist)

        dist = cp.where(update_mask, distances_from_u, dist)
        parent = cp.where(update_mask, u, parent)

    print(f"Total MST Cost: {mst_cost}")
    return mst_edges

def create_mst_cpu_prim(nodes):
    num_nodes = len(nodes)
    if num_nodes == 0:
        return []
    
    nodes_gpu = np.asarray(nodes, dtype=np.float32)
    
    dist = np.full(num_nodes, np.inf, dtype=np.float32)

    parent = np.full(num_nodes, -1, dtype=np.int32)

    visited = np.zeros(num_nodes, dtype=np.bool_)

    dist[0] = 0
    
    mst_edges = []
    mst_cost = 0

    for _ in trange(num_nodes, desc="Building MST with Prim's"):
        temp_dist = np.where(visited, np.inf, dist)
        u = int(np.argmin(temp_dist))

        if np.isinf(dist[u]):
            break

        visited[u] = True

        if parent[u] != -1:
            p = int(parent[u])
            d = float(dist[u])
            mst_edges.append((p, u, d))
            mst_cost += d

        distances_from_u = np.sqrt(np.sum((nodes_gpu - nodes_gpu[u])**2, axis=1))

        update_mask = (~visited) & (distances_from_u < dist)

        dist = np.where(update_mask, distances_from_u, dist)
        parent = np.where(update_mask, u, parent)

    print(f"Total MST Cost: {mst_cost}")
    return mst_edges

if __name__ == "__main__":
    print("load tsp")
    tsp = TSP('a280.tsp')
    print("start processing")
    st = time.time()
    mst = create_mst_gpu(tsp.node_coord_section)
    et = time.time()
    print(f"processing finished, elasped_time : {et-st}")
    print("\n--- Minimum Spanning Tree (MST) Edges ---")
    print("Format: (Node 1, Node 2, Distance)")
    print(len(mst))
    for i, edge in enumerate(mst[:20]): 
        print(f"Edge {i+1}: ({edge[0]}, {edge[1]}, {edge[2]:.4f})")